# No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models

<p align="center">
  <img src="paper_preview.png" width="600" alt="Paper first page" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>📄 Read the paper on arXiv</strong></a>
</p>

## Abstract

CDD (Contamination Detection via output Distribution) identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models (70M–410M parameters). Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD is unreliable: it performs at chance level in the majority of contaminated conditions, while simpler probability-based methods (perplexity, Min-k% Prob) consistently detect the contamination. CDD only succeeds when training produces memorization strong enough to collapse the output distribution — a condition that parameter-efficient fine-tuning often prevents.

## Key Findings

- CDD performs at chance in 22 out of 27 tested conditions (3 FT methods × 3 contamination levels × 3 datasets), while perplexity and Min-k% Prob exceed chance in 24–25 conditions
- Probability-based methods detect contamination even at the lowest contamination level (c=1), where CDD provides zero signal
- The pattern holds across mathematical reasoning (GSM8K), code generation (HumanEval), and competition mathematics (MATH)

## Setup

```bash
# CPU environment
conda env create -f environment.yml

# GPU environment (CUDA 12.4, for training and detection)
conda env create -f environment_gpu.yml
```

## Project Structure

```
contamination_detection/          # Core library
  data/                           # Loading, splitting, formatting, contamination injection
  detection/                      # CDD: sampling, edit distance, peakedness, classification
  training/                       # Model loading (LoRA + full FT), fine-tuning
  baselines/                      # Perplexity, Min-k% Prob, n-gram overlap, random
  evaluation/                     # Metrics, confidence intervals, significance tests
  visualization/                  # Plotting utilities
  analysis/                       # Scale analysis

configs/                          # Hydra configuration files
tests/                            # Unit tests
```

## Experimental Setup

- **Models**: Pythia-70M, Pythia-160M, Pythia-410M (EleutherAI)
- **Datasets**: GSM8K (500 examples), HumanEval (164 examples), MATH (500 examples)
- **Fine-tuning**: LoRA r=8, LoRA r=256, full fine-tuning; 3 and 20 epochs
- **Contamination levels**: 0, 1, 5, 10 repetitions of leaked data
- **Detection methods**: CDD, perplexity, Min-k% Prob, n-gram overlap

## Citation

```bibtex
@misc{sela2026nomemorizationnodetection,
  title={No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models},
  author={Sela, Omer},
  year={2026},
  url={https://github.com/Sela-Omer/Contamination-Detection-Small-LM}
}
```

## References

- Dong et al. "[Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models](https://arxiv.org/abs/2402.15938)." Findings of ACL 2024.
- Shi et al. "[Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789)." ICLR 2024.
- Biderman et al. "[Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/abs/2304.01373)." ICML 2023.
- Cobbe et al. "[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)." 2021.
- Chen et al. "[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)." 2021.
- Hendrycks et al. "[Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)." NeurIPS 2021.
- Hu et al. "[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)." ICLR 2022.
