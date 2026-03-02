# No Memorization, No Detection

**Output Distribution-Based Contamination Detection in Small Language Models**

Omer Sela | Tel Aviv University | NLP Course Final Project (2025)

## Overview

This project investigates whether CDD (Contamination Detection via output Distribution) from [Dong et al. (ACL Findings 2024)](https://arxiv.org/abs/2402.15938) works on small language models (70M-410M parameters). We find that CDD detects **memorization**, not contamination per se. When parameter-efficient fine-tuning limits a model's ability to memorize specific training examples, CDD fails silently even on heavily contaminated data.

## Key Finding

With LoRA r=8 and 3 training epochs, CDD performs at chance level (50%) across all model sizes, even when n-gram overlap confirms the data is contaminated. Only when fine-tuning produces sufficient memorization (through high LoRA rank, full fine-tuning, or extended training) does CDD recover strong detection accuracy (up to 95.5%).

## Setup

```bash
# Create conda environment
conda env create -f environment.yml        # CPU (Mac)
conda env create -f environment_gpu.yml    # GPU (CUDA 12.4)

# Activate
conda activate contamination-detection     # or 'cdd' for GPU env
```

## Project Structure

```
contamination_detection/    # Core library
  data/                     # Data loading, splitting, formatting, contamination
  detection/                # CDD: sampler, edit distance, peakedness, classifier
  training/                 # Model loading (LoRA + full), fine-tuning
  baselines/                # Random, perplexity, n-gram baselines
  evaluation/               # Metrics, confidence intervals, significance tests
  visualization/            # Plotting utilities
  analysis/                 # Scale analysis

scripts/                    # Experiment scripts
  prepare_data.py           # Create data splits (run once)
  run_single_condition.py   # Run one model x contam x ft_method condition
  launch_parallel.sh        # Launch all conditions across GPUs
  launch_lora256.sh         # LoRA r=256 experiments
  launch_full.sh            # Full fine-tuning experiments
  launch_lora8_ep20.sh      # LoRA r=8, 20 epochs
  launch_lora256_ep20.sh    # LoRA r=256, 20 epochs
  launch_full_ep20.sh       # Full fine-tuning, 20 epochs
  generate_paper_figures_v3.py  # Publication figures

configs/                    # Hydra configuration files
tests/                      # Unit tests
doc/latex/                  # ACL-format paper (LaTeX)
```

## Running Experiments

```bash
# 1. Prepare shared data splits
python scripts/prepare_data.py --output_dir outputs/gpu_full_run

# 2. Run a single condition
CUDA_VISIBLE_DEVICES=0 python scripts/run_single_condition.py \
    --model EleutherAI/pythia-410m \
    --contam_epochs 10 \
    --ft_method full \
    --train_epochs 3 \
    --output_dir outputs/gpu_full_run

# 3. Or launch all conditions in parallel
bash scripts/launch_parallel.sh
```

## Models and Dataset

- **Models**: Pythia-70M, Pythia-160M, Pythia-410M (EleutherAI)
- **Dataset**: GSM8K (500 examples: 300 train, 100 contamination, 100 evaluation)
- **Fine-tuning**: LoRA r=8, LoRA r=256, full fine-tuning; 3 and 20 epochs
- **Contamination levels**: 0, 1, 5, 10 repetitions of leaked data

## Citation

```bibtex
@misc{sela2025memorization,
  title={No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models},
  author={Sela, Omer},
  year={2025},
  note={Tel Aviv University, NLP Course Final Project}
}
```

## References

- Dong et al. "Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models." ACL Findings 2024.
- Biderman et al. "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling." ICML 2023.
- Cobbe et al. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
