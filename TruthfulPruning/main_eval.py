import torch
import argparse
import numpy as np
from huggingface_hub import login
from accelerate.logging import get_logger
from transformers.utils.versions import require_version
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaTokenizer, PreTrainedTokenizerFast)

# Local files
from lib.prune_all import check_sparsity
from lib.eval import eval_ppl, eval_acc, eval_bleu
from lib.eval import run_benchmarking, evaluate_task_result

print('\n=>=> # of gpus: ', torch.cuda.device_count())

"""from accelerate.logging import get_logger"""
logger = get_logger(__name__)
"""from transformers.utils.versions import require_version"""
require_version("datasets>=1.8.0",
                "To fix: pip install "
                "-r examples/pytorch/language-modeling/requirements.txt")


ppl_datasets = ('wikitext', 'redpajama', 'oscar')
acc_datasets = ('gsm8k', 'svamp', 'mawps', 'esnli', 'anli_r1',
                'anli_r2', 'anli_r3', 'commonsense_qa',
                'race', 'winogrande')
bleu_datasets = ('wmt14', 'iwslt')

acc_datasets = ('svamp', 'mawps', 'commonsense_qa','winogrande')
acc_datasets = ('svamp', 'mawps', 'commonsense_qa', 'winogrande')


def get_llm(model_path):
    if "llama" in model_path or "Llama" in model_path:
        print("=>=> LlamaForCausalLM...")
        model = LlamaForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir='./llm_weights')
                                                 #device_map="auto",
                                                 #low_cpu_mem_usage=True)
    else:
        print("=>=> AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     cache_dir='./llm_weights')
    model.seqlen = 2048


    if "Mistral" in model_path:
        print("\n=>=> Triggering AutoTokenizer...\n")
        tokenizer = (
            AutoTokenizer.from_pretrained(model_path, use_fast=False)
            )
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    elif "llama" in model_path or "Llama" in model_path:
        if "Llama-3" in model_path or "llama3" in model_path:
            print("\n=>=> Triggering PreTrainedTokenizerFast...\n")
            tokenizer = (
                PreTrainedTokenizerFast.from_pretrained(model_path,
                                                        use_fast=False)
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        else:
            print("\n=>=> Triggering LlamaTokenizer...\n")
            tokenizer = (
                LlamaTokenizer.from_pretrained(model_path,
                                               use_fast=False)
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    else:
        print("Model {} isn't supported!".format(model_path))
        exit(-1)
    # Ensure `pad_token_id` is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Length of context window in tokens')
    parser.add_argument('--input_format',
                        type=str, default='concat',
                        choices=['autoregressive', 'single',
                                 'concat', 'zero'])
    parser.add_argument('--shot', type=str,
                        default='few', choices=['few', 'zero'])
    parser.add_argument('--padding_side', type=str,
                        default='left', choices=['left', 'right'])
    parser.add_argument('--num_incontext', type=int, 
                        help='Number of in-context Q-A pairs in '
                             'each calibration sample.')
    parser.add_argument('--rationale', action='store_true', 
                        help='If flag is included, include CoT rationale '
                        'in answer portion of Q-A pairs in calibration samples.')
    parser.add_argument('--num_cot_steps', type=int, 
                        help='Number of CoT reasoning steps for each '
                             'Q-A pair in calibration samples. Only '
                             'used if `--rationale` is included.')
    parser.add_argument('--eval_rationale', action='store_true', default=True,
                        help='If flag is included, at evaluation time, '
                             'include CoT rationale in in-context examples '
                             'in prompt.')
    args = parser.parse_args()
    """
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf'
    'llama3_instruct_8B': 'Undi95/Meta-Llama-3-8B-Instruct-hf'
    'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3'
    """
    # model_path = 'pruned_0.7llama7B'
    model_path = '/home/yaofu/yxf484/Pruning/TruthfulPruning/pruned_model/llama3_instruct_8b/c4_owl_option2_0.5'
    model_name = 'llama3'
    cuda_id = 2
    sparsity_ratio = 0.5
    verbose = True

    """
       If flag is included, run the benchmark task.
    ['boolq', 'rte', 'hellaswag', 'winogrande',
     'openbookqa', 'arc_easy', 'arc_challenge']
    """
    run_benchmark = False
    benchmark_task = 'boolq'

    """
       If flag is included, at evaluation time, 
    include CoT rationale in in-context examples 
    in prompt.
    ['wikitext', 'redpajama', 'oscar', 'gsm8k',
     'svamp', 'mawps', 'anli_r1', 'anli_r2',
     'anli_r3', 'esnli', 'rte', 'boolq',
     'commonsense_qa', 'race', 'winogrande',
     'wmt14', 'iwslt', 'all']
    """
    eval_rationale = False
    eval_dataset = 'all'


    """from huggingface_hub import login"""
    login(token="")
    print("=>=> Number of calibration samples is {}".format(args.nsamples))
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)


    print("\n=>=> Loading LLM model from {}...".format(model_path))
    model, tokenizer = get_llm(model_path=model_path)
    print("=>=> Finish Loading the LLM model from {}!\n".format(model_path))
    print("\n=>=> Model is ====================================================")
    print("=>=> model.__class__.__name__ is {}"
          .format(model.__class__.__name__))
    print("\n=>=> model is")
    print(model)
    print("\n")
    model.eval()

    

    cuda_name = "cuda:" + str(cuda_id)
    device = torch.device(cuda_name)
    model = model.to(device)
    if "30b" in model_path or "65b" in model_path:
        print("\n=>=> Your model size is >= 30B...\n")
        # for 30b and 65b we use device_map to load onto
        # multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("=>=> Use device ", device)
    print("=>=> Target sparsity", sparsity_ratio)

    ################################################################
    print("=>=> Start checking the sparsity...")
    print("\n\n")
    print("*" * 30)
    """from lib.prune_all import check_sparsity"""
    sparsity_ratio = check_sparsity(model=model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    print("=>=> Finish checking the sparsity!")
    print("\n\n")
    ################################################################
    print("\n=>=> Start evaluation...")
    if run_benchmark:
        """from lib.eval import run_benchmarking"""
        print('\n=>=> Running evaluation...\n')
        run_benchmarking(model, tokenizer, benchmark_task)
        print('\n =>=> Evaluating on benchmarks...\n')
        """from lib.eval import evaluate_task_result"""
        evaluate_task_result(benchmark_task)
        print('\n =>=> Finish benchmark evaluation!\n')
        exit(0)
    if eval_dataset == 'all':
        print("\n")
        """
        for dataset in ppl_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            # from lib.eval import eval_ppl
            ppl = eval_ppl(args, dataset, model, tokenizer, device,
                           verbose=verbose)
            print(f"=>=> Perplexity on {dataset}: {ppl}")
        print("\n")
        """
        for dataset in acc_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            """from lib.eval import eval_acc"""
            acc = eval_acc(args, model, tokenizer, device, dataset=dataset,
                           shot=args.shot, verbose=verbose)
            print(f"=>=> Acc. on {dataset}: {acc}")
        print("\n")
        """
        for dataset in bleu_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            # from lib.eval import eval_bleu
            bleu = eval_bleu(args, model, tokenizer, device,
                             dataset=dataset,
                             verbose=verbose)['bleu']
            print(f"=>=> BLEU on {dataset}: {bleu}")
        """
    elif eval_dataset in ppl_datasets:
        print("\n")
        """from lib.eval import eval_ppl"""
        ppl = eval_ppl(model, tokenizer, device, verbose=verbose)
        print(f"=>=> Perplexity on {args.eval}: {ppl}")
    elif eval_dataset in acc_datasets:
        print("\n")
        """from lib.eval import eval_acc"""
        acc = eval_acc(args, model, tokenizer, device, dataset=eval_dataset,
                       shot=args.shot, verbose=args.verbose)
        print(f"=>=> Acc. on {args.eval}: {acc}")
    elif eval_dataset in bleu_datasets:
        print("\n")
        """from lib.eval import eval_bleu"""
        bleu = eval_bleu(args, model, tokenizer, device,
                         dataset=eval_dataset,
                         shot=args.shot,
                         verbose=args.verbose)['bleu']
        print(f"=>=> bleu on {args.eval}: {bleu}")
    else:
        print("\n")
        print("=>=> No evaluation...")
    print("=>=> Finish evaluation!\n")

    
if __name__ == '__main__':
    main()
