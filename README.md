# CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and  Selective Sparsification

📃Junhui He, Shangyu Wu, Weidong Wen, Chun Jason Xue, Qingan Li: CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification. **EMNLP 2024 Main**: 18658-18668 🔗[Link](https://aclanthology.org/2024.emnlp-main.1038/)

## Requirements

- Miniconda (recommended)
- NVIDIA GPU with at least 24GB VRAM for threshold computation

## Install Dependencies

1. Clone the repository:

    ```bash
    git clone https://github.com/junhuihe-hjh/CHESS.git --recursive
    cd CHESS
    ```

2. Install dependencies:

    ```bash
    # Create miniconda environment (recommended)
    conda create --name CHESS "python<3.13"
    conda activate CHESS

    # Install
    pip install -r requirements.txt
    pip install -e ./lm-evaluation-harness
    ```

## Run Performance Benchmarks

1. Download the C4 dataset and the Llama-3.1-8B model:

    ```bash
    huggingface-cli download allenai/c4 --local-dir huggingface-datasets/c4 --include en/c4-train.00000-of-01024.json.gz --repo-type dataset
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir huggingface-models/Llama-3.1-8B-Instruct --exclude original/*
    ```

2. Compute and save thresholds: 

    ```bash
    cd ./CHESS/notebooks
    python thresholds.py
    cd ../..
    ```

    Thresholds will be written to `./CHESS/thresholds/0_5.pt`

3. Run benchmarks on downstream tasks:

    ```bash
    cd ./benchmark
    ./run.sh
    ```