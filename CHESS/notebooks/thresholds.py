# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
import torch.functional as F
import types
from tqdm import tqdm
from typing import *
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
import transformers.models.llama.modeling_llama as modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

# %%
model_dir = r'../../huggingface-models/Llama-3.1-8B-Instruct'

# %%
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to('cuda')

# %%
dataset_dir = r'../../huggingface-datasets/c4'
dataset = datasets.load_dataset(dataset_dir, data_files='en/c4-train.00000-of-01024.json.gz')

# %%
dtype = torch.bfloat16
device = 'cuda'

ffn_size = model.config.intermediate_size
num_layers = model.config.num_hidden_layers

# %%
attention_inputs_thresholds = [torch.zeros([1,], dtype=dtype, device=device) for _ in range(num_layers)]
attention_outputs_thresholds = [torch.zeros([1,], dtype=dtype, device=device) for _ in range(num_layers)]

gate_proj_states_thresholds_cats = [torch.zeros([1,], dtype=dtype, device=device) for _ in range(num_layers)]

up_proj_states_abs_mean = [torch.zeros(ffn_size, dtype=dtype, device=device) for _ in range(num_layers)]
importance_thresholds = [torch.zeros([1,], dtype=dtype, device=device) for _ in range(num_layers)]
gate_proj_states_thresholds_chess = [torch.zeros(ffn_size, dtype=dtype, device=device) for _ in range(num_layers)]

# %%
sparsity = 0.5
calibration_set_size = 64 * 1024

# %%
layer_idx = None

# %%
def quantile(x, q, dim=None):
    if dim is None:
        num_elements = x.numel()
    else:
        num_elements = 1
        for dim_idx in range(len(x.shape)):
            if dim_idx != dim:
                num_elements *= x.shape[dim_idx]
    
    k = int(num_elements * q)

    if dim is None:
        return torch.kthvalue(x.view(-1), k, dim=-1).values
    else:
        return torch.kthvalue(x, k, dim=dim).values

# %%
def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    global layer_idx
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    layer_idx = self.layer_idx
    attention_inputs_thresholds[layer_idx] += quantile(hidden_states.abs(), q=sparsity)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attention_outputs_thresholds[layer_idx] += quantile(attn_output.abs(), q=sparsity)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

# %%
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(llama_attention_forward, layer.self_attn)

# %%
def llama_mlp_forward(self, x):
    gate_proj_states = self.act_fn(self.gate_proj(x))
    up_proj_states = self.up_proj(x)
    down_proj = self.down_proj(gate_proj_states * up_proj_states)

    ffn_size = self.config.intermediate_size

    gate_proj_states_thresholds_cats[layer_idx] += quantile(gate_proj_states.abs(), q=sparsity)
    up_proj_states_abs_mean[layer_idx] += up_proj_states.abs().view(-1, ffn_size).mean(dim=0)

    return down_proj

# %%
for layer in model.model.layers:
    layer.mlp.forward = types.MethodType(llama_mlp_forward, layer.mlp)

# %%
max_len = 64 * 1024
num_processed_tokens = 0

with torch.no_grad():
    with tqdm(total=calibration_set_size) as pbar:
        for sample_idx in range(len(dataset['train'])):
            input_ids = tokenizer.encode(dataset["train"][sample_idx]['text'])
            input_ids = input_ids[:max_len]

            model(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device))
            
            pbar.update(len(input_ids))
            
            num_processed_tokens += len(input_ids)
            if num_processed_tokens >= calibration_set_size:
                break

num_samples = sample_idx + 1

# %%
for layer_idx in range(num_layers):
    attention_inputs_thresholds[layer_idx] /= num_samples
    attention_outputs_thresholds[layer_idx] /= num_samples
    gate_proj_states_thresholds_cats[layer_idx] /= num_samples
    up_proj_states_abs_mean[layer_idx] /= num_samples

# %%
layer_idx = 0

# %%
for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(modeling_llama.LlamaAttention.forward, layer.self_attn)

# %%
def llama_mlp_forward(self, x):
    global layer_idx
    
    gate_proj_states = self.act_fn(self.gate_proj(x))
    up_proj_states = self.up_proj(x)
    down_proj = self.down_proj(gate_proj_states * up_proj_states)

    importance_scores = gate_proj_states.view(-1, ffn_size).abs() * up_proj_states_abs_mean[layer_idx].to(device)
    importance_thresholds[layer_idx] += quantile(importance_scores, q=sparsity)
    layer_idx = (layer_idx + 1) % num_layers

    return down_proj

# %%
for layer in model.model.layers:
    layer.mlp.forward = types.MethodType(llama_mlp_forward, layer.mlp)

# %%
max_len = 64 * 1024
num_processed_tokens = 0

with torch.no_grad():
    with tqdm(total=calibration_set_size) as pbar:
        for sample_idx in range(len(dataset['train'])):
            input_ids = tokenizer.encode(dataset["train"][sample_idx]['text'])
            input_ids = input_ids[:max_len]

            model(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device))
            
            pbar.update(len(input_ids))
            
            num_processed_tokens += len(input_ids)
            if num_processed_tokens >= calibration_set_size:
                break

num_samples = sample_idx + 1

# %%
for layer_idx in range(num_layers):
    importance_thresholds[layer_idx] /= num_samples
    gate_proj_states_thresholds_chess[layer_idx] = importance_thresholds[layer_idx] / up_proj_states_abs_mean[layer_idx]

# %%
import os

output_dir = r'../thresholds'
os.makedirs(output_dir, exist_ok=True)

torch.save({
    'attention_inputs_thresholds': attention_inputs_thresholds,
    'attention_outputs_thresholds': attention_outputs_thresholds,
    'gate_proj_states_thresholds_cats': gate_proj_states_thresholds_cats,
    'gate_proj_states_thresholds_chess': gate_proj_states_thresholds_chess,
}, f'{output_dir}/0_5.pt')


