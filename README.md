# Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs (2025-EMNLP-Findings-Poster)

## 🚀 Step 1: Truthful Pruning, OWL Pruning, and Wanda Pruning

📦 **Installation** 

```bash
cd TruthfulPruning

conda create -n truthful_pruning python=3.9

conda activate truthful_pruning

pip install -r requirements.txt
```

📦 **Quickstart**

We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. 
The default is `llm_weights`.
- `--prune_method`: Pruning methods,namely [`wanda`,`wanda_option1`, `owl`,`owl_option2`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--save`: Specifies the directory where the result will be stored.
- `--Hyper_m`: Denotes the hyperparameter of `M` for OWL pruning.
- `--Lamda`:  Denotes the hyperparameter of `Lamda` for OWL pruning.

**Models**
- 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
- 'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
- 'llama3_instruct_8B': 'Undi95/Meta-Llama-3-8B-Instruct-hf',
- 'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'

---

**Script example of pruning via vanilla wanda and C4**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets c4 \
--input_format concat \
--nsamples 128 --seqlen 2048 \
--padding_side left \
--prune_method wanda \
--sparsity_ratio 0.5 \
--sparsity_type unstructured
```

**Script example of pruning via vanilla wanda and enriched truthfulQA**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets enriched_truth_qa \
--input_format concat \
--nsamples 128 --seqlen 2048 \
--padding_side left \
--prune_method wanda \
--sparsity_ratio 0.5 \
--sparsity_type unstructured
```

**Script example of pruning via wanda option 1 and C4**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets c4 \
--input_format concat \
--nsamples 128 --seqlen 2048 \
--padding_side left \
--prune_method wanda_option1 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--skip_dense_eval
```

**Script example of pruning via OWL and C4**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets c4 \
--prune_method owl \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--skip_dense_eval
```

**Script example of pruning via OWL option2 and C4**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets c4 \
--prune_method owl_option2 \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--skip_dense_eval
```

**Script example of pruning via OWL option2 and C4, enriched TruthfulQA**
```
python3 main.py \
--cuda_id 0 \
--model Undi95/Meta-Llama-3-8B-Instruct-hf \
--calibration_datasets c4 enriched_truth_qa \
--input_format concat \
--nsamples 128 --seqlen 2048 \
--padding_side left \
--prune_method owl \
--sparsity_ratio 0.5 \
--sparsity_type unstructured \
--Lamda 0.08 \
--Hyper_m 5 \
--skip_dense_eval
```

## 🚀 Step 2: Lie Detection

📦 **Installation** 

```bash
cd Truth_is_Universal

conda create --name truth_is_universal python=3.11

conda activate truth_is_universal

pip install -r requirements.txt
```

📦 **Quickstart**

**Generate the intermediate activations:**
```bash
python3 generate_acts.py
```
**Generate the Figure 1 (Separation between true and false statements across layers):**
```bash
python3 truth_directions_main.py
python3 truth_directions_main_comparison.py
```
**Do the lie detection:**
```bash
python3 lie_detection_main.py
```  

## 📚 Citation

If you find our work helpful, please consider citing:
```bibtex
@article{fu2025pruning,
  title={Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs},
  author={Fu, Yao and Li, Runchao and Long, Xianxuan and Yu, Haotian and Han, Xiaotian and Yin, Yu and Li, Pan},
  journal={arXiv preprint arXiv:2509.00096},
  year={2025}
}
```

## 🙏 Acknowledgement
Our codes are built upon [OWL](https://github.com/luuyin/OWL) and [Truth_is_Universal](https://github.com/sciai-lab/Truth_is_Universal).
