import sys
import torch
import argparse
import numpy as np
from huggingface_hub import login
from accelerate.logging import get_logger
from transformers import MODEL_MAPPING, SchedulerType
from transformers.utils.versions import require_version
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM,
                          LlamaTokenizer, PreTrainedTokenizerFast)

### Local files
from lib.prune_all import check_sparsity
from lib.eval import eval_ppl, eval_acc, eval_bleu
from lib.eval import run_benchmarking, evaluate_task_result
from lib.prune_all import (prune_wanda, prune_wanda_option1,
                           prune_owl, prune_owl_option2)


login(token="")

print('\n=>=> # of gpus: ', torch.cuda.device_count())
"""
   from transformers import MODEL_MAPPING
   MODEL_MAPPING is a dictionary-like object that maps model configurations 
(like BERTConfig, GPT2Config) to their respective model classes.
   MODEL_CONFIG_CLASSES is now a list containing all the model configuration classes 
available in MODEL_MAPPING's keys (which are the model configuration classes, like 
BertConfig, GPT2Config, etc.).
   MODEL_TYPES is a tuple of the string representations of the model types 
(e.g., 'bart', 'bert', 'bert-generation', 'big_bird') extracted from 
MODEL_CONFIG_CLASSES.
"""
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
# print('\n=>=> MODEL_CONFIG_CLASSES: {}\n'.format(MODEL_CONFIG_CLASSES))
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# print('\n=>=> MODEL_TYPES: {}\n'.format(MODEL_TYPES))

"""from accelerate.logging import get_logger"""
logger = get_logger(__name__)
"""from transformers.utils.versions import require_version"""
require_version("datasets>=1.8.0",
                "To fix: pip install "
                "-r examples/pytorch/language-modeling/requirements.txt")

# ppl_datasets = ('oscar', 'redpajama', 'wikitext2')
# ppl_datasets = ('redpajama', 'wikitext2')
ppl_datasets = ('wikitext2')
acc_datasets = ('gsm8k', 'svamp', 'mawps', 'esnli', 'anli_r1',
                'anli_r2', 'anli_r3', 'commonsense_qa',
                'race', 'winogrande')
bleu_datasets = ('wmt14', 'iwslt')


def get_llm(model, cache_dir, seqlen):
    if "llama" in model or "Llama" in model:
        print("=>=> LlamaForCausalLM...")
        model = LlamaForCausalLM.from_pretrained(model, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir=cache_dir)
                                                 #device_map="auto",
                                                 #low_cpu_mem_usage=True)
    else:
        print("=>=> AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.float16,
                                                     cache_dir=cache_dir)

    
    print("=> model.config.max_position_embeddings is {}"
          .format(model.config.max_position_embeddings))
    
    model.seqlen = seqlen
    print("=> model.seqlen is {}".format(model.seqlen))
    print("=> model.config.max_position_embeddings is {}"
          .format(model.config.max_position_embeddings))

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--model', type=str, help='Model path',
                        default='luodian/llama-7b-hf')
    parser.add_argument("--cache_dir", default="llm_weights", type=str,
                        help="Directory for locally loading or storing LLMs.")

    ## Calibration data
    parser.add_argument('--calibration_datasets', type=str,
                        nargs='+', default='c4',
                        choices=['c4', 'pile', 'oscar', 'redpajama',
                                 'gsm8k', 'svamp', 'mawps',
                                 'esnli', 'anli_r1', 'anli_r2', 'anli_r3',
                                 'rte', 'boolq', 'commonsense_qa',
                                 'race', 'winogrande', 'wmt14', 'iwslt',
                                 'ellipses', 'random', 'enriched_truth_qa',
                                 'truth_is_universal'],
                        help='1 Pretraining datasets (C4: c4, Pile: pile, OSCAR: oscar, RedPajama: redpajama) '
                             '2 Downstream datasets: '
                             '2.1 Arithmetic reasoning (GSM8K: gsm8k, SVAMP: svamp, MAWPS: mawps)'
                             '2.2 Natural language inference (e-SNLI: esnli, ANLI: anli_r1, ANLIR3: anli_r3)'
                             '2.3 Commonsense reasoning (CommonsenseQA: commonsense_qa, '
                             'RACE: race, WinoGrande: winogrande)'
                             '3 Nonsense data (ellipses: ellipses, random alphanumeric strings: random')
    parser.add_argument('--input_format',
                        type=str, default='concat',
                        choices=['autoregressive', 'single',
                                 'concat', 'zero'])
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Length of context window in tokens')
    parser.add_argument('--padding_side', type=str,
                        default='left', choices=['left', 'right'])
    parser.add_argument('--shot', type=str,
                        default='few', choices=['few', 'zero'])
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

    ## Pruning hyperparameters
    parser.add_argument('--sparsity_ratio', type=float, default=0,
                        help='Sparsity level, denoting the percentage of '
                             'weights to be pruned.')
    parser.add_argument("--prune_method", type=str,
                        help="Pruning methods,namely [owl, "
                             "wanda_owl_structure, sparsegpt_owl, "
                             "magnitude, wanda, sparsegpt]")
    parser.add_argument('--use_variant', action="store_true",
                        help="Whether to use the wanda variant described "
                             "in the appendix")
    parser.add_argument("--sparsity_type", type=str,
                        default='unstructured',
                        help='For the initialization of prune_n, prune_m '
                             'if structured')
    parser.add_argument("--Lamda", default=0.08, type=float,
                        help="Denotes the hyperparameter of `Lamda`")
    parser.add_argument('--Hyper_m', type=float, default=5,
                        help="Denotes the hyperparameter of M")
    parser.add_argument("--outlier_by_activation",
                        action="store_true", help="outlier_by_activation")
    parser.add_argument("--outlier_by_wmetric", action="store_true",
                        help="outlier_by_wmetric")

    ## Evaluation
    parser.add_argument('--verbose', action='store_true',
                        help='If this flag is included, print '
                             'intermediate results to stdout.')
    parser.add_argument('--skip_dense_eval',
                        action='store_true',
                        help='If flag is included, skip evaluation '
                             'of dense model (before pruning).')
    parser.add_argument('--run_benchmark',
                        action='store_true',
                        help='If flag is included, run the benchmark task.')
    parser.add_argument('--benchmark_task',
                        type=str, default='boolq',
                        choices=['boolq', 'rte', 'hellaswag', 'winogrande',
                                 'openbookqa', 'arc_easy', 'arc_challenge'])
    parser.add_argument('--eval_rationale', action='store_true',
                        help='If flag is included, at evaluation time, '
                             'include CoT rationale in in-context examples '
                             'in prompt.')
    parser.add_argument('--eval', type=str, default='wikitext2',
                        choices=['wikitext2', 'redpajama', 'oscar', 'gsm8k',
                                 'svamp', 'mawps', 'anli_r1', 'anli_r2',
                                 'anli_r3', 'esnli', 'rte', 'boolq',
                                 'commonsense_qa', 'race', 'winogrande',
                                 'wmt14', 'iwslt', 'all', 'all_ppl'])

    ## Results saving
    parser.add_argument('--save', type=str, default='save_test/',
                        help='Specifies the directory where the result '
                             'will be stored.')
    parser.add_argument('--save_model', type=str,
                        default='pruned_model/',
                        help='Path to save the pruned model.')


    args = parser.parse_args()


    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, \
            "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        print("\n=>=> args.sparsity_type is {}, "
              "prune_n is {}, prune_m is {}"
              .format(args.sparsity_type,
                      prune_n, prune_m))
    
    
    model_name = args.model.split("/")[-1]
    print("\n=>=> Loading LLM model from {}...".format(args.model))
    model = get_llm(model=args.model, cache_dir=args.cache_dir,
                    seqlen=args.seqlen)
    print("=>=> Finish Loading the LLM model from {}!\n"
          .format(args.model))
    print("\n=>=> Model is ===============================================")
    print("=>=> model.__class__.__name__ is {}"
          .format(model.__class__.__name__))
    print("\n=>=> model is")
    print(model)
    print("\n")

    """
    1. Disabling Dropout:
    • In evaluation mode (model.eval()), dropout is disabled, meaning 
    all neurons are active, allowing the model to use its full capacity 
    and give stable predictions.
    2. Batch Normalization:
    • If the model has batch normalization layers (common in many NNs), 
    they behave differently in evaluation mode. In training mode, batch 
    normalization computes the mean and variance of the current batch to 
    normalize the data. In evaluation mode, it uses the running estimates 
    of mean and variance that were computed during training, so that the 
    output is stable.
    3. Inference Mode:
    • model.eval() is essential when you want to perform inference or 
    evaluation, because it ensures that the model behaves consistently 
    and that the stochastic behaviors (like dropout) used during training 
    are turned off.
    4. No Gradient Updates:
    • While model.eval() doesn’t by itself affect gradient calculations, 
    during evaluation, it’s common to also use torch.no_grad() in conjunction 
    with it. This explicitly tells PyTorch not to compute gradients, which 
    saves memory and improves inference speed.
    """
    cuda_name = "cuda:" + str(args.cuda_id)
    device = torch.device(cuda_name)
    print("=>=> Use device {}".format(device))
    print("=>=> Target sparsity", args.sparsity_ratio)
    model = model.to(device)
    model.eval()
    

    if "Mistral" in model_name:
        print("\n=>=> Triggering AutoTokenizer...\n")
        tokenizer = (
            AutoTokenizer.from_pretrained(args.model, use_fast=False,
                                          padding_side=args.padding_side))
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    elif "llama" in model_name or "Llama" in model_name:
        if "Llama-3" in model_name:
            print("\n=>=> Triggering PreTrainedTokenizerFast...\n")
            tokenizer = (
                PreTrainedTokenizerFast.from_pretrained(args.model,
                                                        use_fast=False,
                                                        padding_side=args.padding_side)
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        else:
            print("\n=>=> Triggering LlamaTokenizer...\n")
            tokenizer = (
                LlamaTokenizer.from_pretrained(args.model,
                                               use_fast=False,
                                               padding_side=args.padding_side)
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    else:
        print("Model {} isn't supported!".format(model_name))
        exit(-1)


    if "30b" in args.model or "65b" in args.model:
        print("\n=>=> Your model size is >= 30B...\n")
        # for 30b and 65b we use device_map to load onto
        # multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]


    if not args.skip_dense_eval:
        if args.run_benchmark:
            """from lib.eval import run_benchmarking"""
            print('\n=>=> Running benchmarking...\n')
            run_benchmarking(model, tokenizer, args.benchmark_task)
            print('\n =>=> Evaluating on benchmarks...\n')
            """from lib.eval import evaluate_task_result"""
            evaluate_task_result(args.benchmark_task)
            print('\n =>=> Finish benchmark evaluation!\n')
        print('\n=>=> Dense Model Scores:')
        if args.eval == 'all':
            print("\n")
            for dataset in ppl_datasets:
                print(f"\n=>=> Begin evaluating on {dataset}")
                """from lib.eval import eval_ppl"""
                ppl = eval_ppl(args, dataset, model, tokenizer, device,
                               verbose=args.verbose)
                print(f"=>=> Perplexity on {dataset}: {ppl}")
            print("\n")
            for dataset in acc_datasets:
                print(f"\n=>=> Begin evaluating on {dataset}")
                """from lib.eval import eval_acc"""
                acc = eval_acc(args, model, tokenizer, device,
                               dataset=dataset,
                               verbose=args.verbose)
                print(f"=>=> Acc. on {dataset}: {acc}")
            print("\n")
            for dataset in bleu_datasets:
                print(f"\n=>=> Begin evaluating on {dataset}")
                """from lib.eval import eval_bleu"""
                bleu = eval_bleu(args, model, tokenizer, device,
                                 dataset=dataset,
                                 verbose=args.verbose)['bleu']
                print(f"=>=> BLEU on {dataset}: {bleu}")
        elif args.eval == 'all_ppl':
            print("\n")
            for dataset in ppl_datasets:
                print(f"\n=>=> Begin evaluating on {dataset}")
                """from lib.eval import eval_ppl"""
                ppl = eval_ppl(args, dataset, model, tokenizer, device,
                               verbose=args.verbose)
                print(f"=>=> Perplexity on {dataset}: {ppl}")
        elif args.eval in ppl_datasets:
            print("\n")
            """from lib.eval import eval_ppl"""
            ppl = eval_ppl(args, args.eval, model, tokenizer, device, verbose=args.verbose)
            print(f"=>=> Perplexity on {args.eval}: {ppl}")
        elif args.eval in acc_datasets:
            print("\n")
            """from lib.eval import eval_acc"""
            acc = eval_acc(args, model, tokenizer, device, dataset=args.eval,
                           shot=args.shot, verbose=args.verbose)
            print(f"=>=> Acc. on {args.eval}: {acc}")
        elif args.eval in bleu_datasets:
            print("\n")
            """from lib.eval import eval_bleu"""
            bleu = eval_bleu(args, model, tokenizer, device,
                             dataset=args.eval,
                             shot=args.shot, verbose=args.verbose)['bleu']
            print(f"=>=> bleu on {args.eval}: {bleu}")
        else:
            print("\n")
            print("=>=> No evaluation...")




    print("=>=> Number of calibration samples is {}".format(args.nsamples))
    print("=>=> Length of the token context window is {}".format(args.seqlen))
    print("\n=>=> The {} pruning starts...".format(args.prune_method))
    if args.prune_method == "wanda":
        """from lib.prune_all import prune_wanda"""
        prune_wanda(args, model, tokenizer, device,
                    prepend_calibration=None,
                    prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_option1":
        """
           from lib.prune_all import prune_wanda_option1
        The pruning ratio be arranged by truthfulness ratio.
        """
        prune_wanda_option1(args, model, tokenizer, device,
                            prepend_calibration=None,
                            prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "owl":
        """from lib.prune_all import prune_wanda_outlier"""
        prune_owl(args, model, tokenizer, device,
                  prepend_calibration=None,
                  prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "owl_option2":
        """from lib.prune_all import prune_wanda_outlier_option2"""
        prune_owl_option2(args, model, tokenizer, device,
                          prepend_calibration=None,
                          prune_n=prune_n, prune_m=prune_m)
    else:
        print("Pruning method {} isn't supported!".format(args.prune_method))
        exit(-1)


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
    if args.run_benchmark:
        """from lib.eval import run_benchmarking"""
        print('\n=>=> Running evaluation...\n')
        run_benchmarking(model, tokenizer, args.benchmark_task)
        print('\n =>=> Evaluating on benchmarks...\n')
        """from lib.eval import evaluate_task_result"""
        evaluate_task_result(args.benchmark_task)
        print('\n =>=> Finish benchmark evaluation!\n')
        print('\n=>=> Dense Model Scores:')
    if args.eval == 'all':
        print("\n")
        for dataset in ppl_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            """from lib.eval import eval_ppl"""
            ppl = eval_ppl(args, dataset, model, tokenizer, device,
                            verbose=args.verbose)
            print(f"=>=> Perplexity on {dataset}: {ppl}")
        print("\n")
        for dataset in acc_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            """from lib.eval import eval_acc"""
            acc = eval_acc(args, model, tokenizer, device,
                            dataset=dataset,
                            verbose=args.verbose)
            print(f"=>=> Acc. on {dataset}: {acc}")
        print("\n")
        for dataset in bleu_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            """from lib.eval import eval_bleu"""
            bleu = eval_bleu(args, model, tokenizer, device,
                             dataset=dataset,
                             verbose=args.verbose)['bleu']
            print(f"=>=> BLEU on {dataset}: {bleu}")
    elif args.eval == 'all_ppl':
        print("\n")
        for dataset in ppl_datasets:
            print(f"\n=>=> Begin evaluating on {dataset}")
            """from lib.eval import eval_ppl"""
            ppl = eval_ppl(args, dataset, model, tokenizer, device, verbose=args.verbose)
            print(f"=>=> Perplexity on {dataset}: {ppl}")
    elif args.eval in ppl_datasets:
        print("\n")
        """from lib.eval import eval_ppl"""
        ppl = eval_ppl(args, args.eval, model, tokenizer, device, verbose=args.verbose)
        print(f"=>=> Perplexity on {args.eval}: {ppl}")
    elif args.eval in acc_datasets:
        print("\n")
        """from lib.eval import eval_acc"""
        acc = eval_acc(args, model, tokenizer, device, dataset=args.eval,
                       shot=args.shot, verbose=args.verbose)
        print(f"=>=> Acc. on {args.eval}: {acc}")
    elif args.eval in bleu_datasets:
        print("\n")
        """from lib.eval import eval_bleu"""
        bleu = eval_bleu(args, model, tokenizer, device,
                         dataset=args.eval,
                         shot=args.shot,
                         verbose=args.verbose)['bleu']
        print(f"=>=> bleu on {args.eval}: {bleu}")
    else:
        print("\n")
        print("=>=> No evaluation...")
    print("=>=> Finish evaluation!\n")

    sys.stdout.flush()

    # print(f"final ppl on wikitext {ppl}")
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"=>=> The model saved to {args.save_model}")


if __name__ == '__main__':
    main()
