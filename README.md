# No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models

<p align="center">
  <img src="paper_preview.png" width="550" alt="Paper first page" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.03203">
    <img src="https://img.shields.io/badge/arXiv-2603.03203-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv" />
  </a>
</p>

## Abstract

CDD, or Contamination Detection via output Distribution, identifies data contamination by measuring the peakedness of a model's sampled outputs. We study the conditions under which this approach succeeds and fails on small language models ranging from 70M to 410M parameters. Using controlled contamination experiments on GSM8K, HumanEval, and MATH, we find that CDD's effectiveness depends critically on whether fine-tuning produces verbatim memorization. In the majority of conditions we test, CDD performs at chance level even when the data is verifiably contaminated and detectable by simpler methods. We show that probability-based methods, specifically perplexity and Min-k% Prob, outperform CDD in all conditions where any method exceeds chance, suggesting that CDD's peakedness-based approach is insufficient for contamination detection in small language models. Code will be made available upon publication.

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
@misc{sela2026memorizationdetectionoutputdistributionbased,
      title={No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models}, 
      author={Omer Sela},
      year={2026},
      eprint={2603.03203},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.03203}, 
}
```

## 📫 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omer-sela)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=flat&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?hl=en&user=d4wmWAQAAAAJ)
[![Amazon Science](https://img.shields.io/badge/Amazon%20Science-FF9900?style=flat&logo=amazon&logoColor=white)](https://www.amazon.science/author/omer-sela)

## References

- Dong et al. "[Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models](https://arxiv.org/abs/2402.15938)." Findings of ACL 2024.
- Shi et al. "[Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789)." ICLR 2024.
- Biderman et al. "[Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/abs/2304.01373)." ICML 2023.
- Cobbe et al. "[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)." 2021.
- Chen et al. "[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)." 2021.
- Hendrycks et al. "[Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)." NeurIPS 2021.
- Hu et al. "[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)." ICLR 2022.
