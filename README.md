# Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs (2025-EMNLP-Findings-Poster)

## Installation and Quickstart

**Truthful Pruning, OWL Pruning, and Wanda Pruning**

```bash
cd TruthfulPruning

conda create -n truthful_pruning python=3.9

conda activate truthful_pruning

pip install -r requirements.txt
```


**Truthfulness on Imitative Falsehoods**

```bash
cd truthqa_evaluate

conda create -n truthqa_eval python=3.9

conda activate truthqa_eval

pip install -r requirements.txt

pip install -e transformers-4.47.1 (For AWQ)

pip install -e transformers-4.51.3
```


##  Citation

If you want to use our code, please cite as
@article{fu2025pruning,
  title={Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs},
  author={Fu, Yao and Li, Runchao and Long, Xianxuan and Yu, Haotian and Han, Xiaotian and Yin, Yu and Li, Pan},
  journal={arXiv preprint arXiv:2509.00096},
  year={2025}
}
