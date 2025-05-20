from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import types
from typing import *
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

class LLM:
    def __init__(
            self, 
            model_dir: str,
            thresholds_dir: str = None,
            method: str = None,
            sparsity: float = 0.5,
            device: str = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
        ):
        self.device = device
        self.dtype = dtype

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).to(device)
 
        if method is not None:
            assert method in ['CHESS', 'CHESS+', 'CATS']

            thresholds_filename = str(sparsity).replace('.', '_') + '.pt'
            thresholds = torch.load(f'{thresholds_dir}/{thresholds_filename}')
            
            for layer_idx, layer in enumerate(self.hf_model.model.layers):
                layer.self_attn.attention_inputs_thresholds = thresholds['attention_inputs_thresholds'][layer_idx]
                layer.self_attn.attention_outputs_thresholds = thresholds['attention_outputs_thresholds'][layer_idx]
                layer.mlp.gate_proj_states_thresholds_cats = thresholds['gate_proj_states_thresholds_cats'][layer_idx]
                layer.mlp.gate_proj_states_thresholds_chess = thresholds['gate_proj_states_thresholds_chess'][layer_idx]

                layer.self_attn.forward = types.MethodType(llama_attention_forward_wrapper(method), layer.self_attn)
                layer.mlp.forward = types.MethodType(llama_mlp_forward_wrapper(method), layer.mlp)

    def __call__(self, inp):
        return self.hf_model(inp.to(self.device))
        
    def generate(self, input_ids, max_new_tokens=128):
        with torch.no_grad():
            prefill_len = len(input_ids)
            
            kv_cache = DynamicCache()
            
            output = self.hf_model(torch.tensor(input_ids).unsqueeze(0).to(self.device), past_key_values=kv_cache)
            kv_cache = output.past_key_values
            
            input_ids.append(output.logits[0, -1].argmax(dim=-1).item())

            for _ in range(max_new_tokens):
                output = self.hf_model(torch.tensor(input_ids[-1:]).unsqueeze(0).to(self.device), past_key_values=kv_cache)

                kv_cache = output.past_key_values
                input_ids.append(output.logits[0, -1].argmax(dim=-1).item())

                if input_ids[-1] == self.hf_tokenizer.eos_token_id:
                    break
            
            return input_ids[prefill_len:]


def llama_attention_forward_wrapper(method):
    def llama_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # if hidden_states.size(1) == 1:
        if method == 'CHESS+':
            hidden_states = torch.where(hidden_states.abs() < self.attention_inputs_thresholds, 0.0, hidden_states)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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

        # if attn_output.size(1) == 1:
        if method == 'CHESS+':
            attn_output = torch.where(attn_output.abs() < self.attention_outputs_thresholds, 0.0, attn_output)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    return llama_attention_forward

def llama_mlp_forward_wrapper(method):
    def llama_mlp_forward(self, x):
        gate_proj_states = self.act_fn(self.gate_proj(x))
        
        # if x.size(1) ==1:
        if method == 'CHESS' or method == 'CHESS+':
            gate_proj_states = torch.where(gate_proj_states.abs() < self.gate_proj_states_thresholds_chess, 0.0, gate_proj_states)
        elif method == 'CATS':
            gate_proj_states = torch.where(gate_proj_states.abs() < self.gate_proj_states_thresholds_cats, 0.0, gate_proj_states)
            
        down_proj = self.down_proj(gate_proj_states * self.up_proj(x))
        return down_proj

    return llama_mlp_forward

if __name__ == '__main__':
    model_dir = r'/root/workspace/huggingface-models/Llama-3.1-8B-Instruct'
    thresholds_dir = r'/root/workspace/CHESS-v0.2/thresholds'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LLM(model_dir, thresholds_dir, method='CHESS')
    # model = LLM(model_dir)

    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        # {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "Solve this equation: x^2 - 2x - 2 = 0"},
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(tokenizer.decode(input_ids))

    output_ids = model.generate(input_ids, max_new_tokens=1024)

    print(tokenizer.decode(output_ids))