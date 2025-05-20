PYTHON_PATH="../CHESS/models"
THRESHOLDS_DIR="../CHESS/thresholds"
tasks="arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,sciq,winogrande"

lm_eval --model huggingface --model_args pretrained="../huggingface-models/Llama-3.1-8B-Instruct/" --task ${tasks} --output_path results/original
lm_eval --model huggingface --model_args pretrained="../huggingface-models/Llama-3.1-8B-Instruct/",method="CATS",python_path="${PYTHON_PATH}",thresholds_dir="${THRESHOLDS_DIR}" --task ${tasks} --output_path results/CATS
lm_eval --model huggingface --model_args pretrained="../huggingface-models/Llama-3.1-8B-Instruct/",method="CHESS",python_path="${PYTHON_PATH}",thresholds_dir="${THRESHOLDS_DIR}" --task ${tasks} --output_path results/CHESS
lm_eval --model huggingface --model_args pretrained="../huggingface-models/Llama-3.1-8B-Instruct/",method="CHESS+",python_path="${PYTHON_PATH}",thresholds_dir="${THRESHOLDS_DIR}" --task ${tasks} --output_path results/CHESS+