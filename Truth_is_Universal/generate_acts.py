import os
import sys
import glob
import torch
import argparse
import configparser
import pandas as pd

from tqdm import tqdm
from huggingface_hub import login
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          PreTrainedTokenizerFast,
                          LlamaTokenizer, LlamaForCausalLM)
from liger_kernel.transformers import AutoLigerKernelForCausalLM

"""from huggingface_hub import login"""
login(token="")

"""import configparser"""
config = configparser.ConfigParser()
print("\n=>=> You are reading the config file...\nconfig is: ")
config.read('config.ini')
print(config)
print("=>=> Finish reading the config file!")


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args_infor):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs,
                 module_outputs):
        self.out, _ = module_outputs


def load_model(pruning, pruning_ratio, model_family, model_size, model_type):
    if pruning:
        if model_family == "Llama3" and model_size == "8B":
            # model_path = "pruned_model_gate/pruned_model/llama3_instruct_8b/wanda" + str(pruning_ratio)
            """
            model_type_list = ['c4_wanda0.5', 'c4_owl0.5', 'c4_wanda_option1_0.5', 'c4_owl_option2_0.5_0.04', 
                       'c4_truthqa_wanda0.5', 'c4_truthqa_owl0.5', 'c4_truthqa_wanda_option1_0.5', 'c4_truthqa_owl_option2_0.5_0.04']
            """
            model_path = "pruned_model_gate/pruned_model/llama3_instruct_8b/" + model_type
            print("\n=>=> Loading the pruned model from {}\n".format(model_path))
    else:
        model_path = os.path.join(config[model_family]
                                  ['weights_directory'],
                                  config[model_family]
                                  [f'{model_size}_{model_type}_subdir'])
    
    print("\n=>=> Loading from {}\n".format(model_path))
    try:
        if model_family == 'Llama2':
            tokenizer = LlamaTokenizer.from_pretrained(str(model_path))
            # model = LlamaForCausalLM.from_pretrained(str(model_path))
            model = AutoLigerKernelForCausalLM.from_pretrained(str(model_path))
            tokenizer.bos_token = '<s>'
        elif model_family == 'Llama3':
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
            model = AutoLigerKernelForCausalLM.from_pretrained(str(model_path))
            tokenizer.bos_token = '<s>'
        else:
            print("\n=> You are using AutoTokenizer and AutoModelForCausalLM.\n")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            # model = AutoModelForCausalLM.from_pretrained(str(model_path))
            model = AutoLigerKernelForCausalLM.from_pretrained(str(model_path))
        if model_family == "Gemma2":
            """
               Gemma2 requires bfloat16 precision which is only 
            available on new GPUs.
            """
            # Convert the model to bfloat16 precision
            model = model.to(torch.bfloat16)
        else:
            # storing model in float32 precision -> conversion to float16
            model = model.half()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_statements(dataset_name):
    """
       Load statements from csv file,
    return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements


def get_acts(statements, tokenizer, model, layers, device):
    """
       Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)

    # get activations
    acts = {layer: [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement,
                                     return_tensors="pt").to(device)
        model(input_ids)
        """
           model(input_ids) → Forward Pass Only (Logits Prediction)
        • This performs a raw forward pass through the model.
        • It returns the logits (unnormalized probabilities) for each token 
        at each position in the sequence.
        • It does not generate new text but simply predicts the likelihood of 
        the next token at each position in the sequence.
           Typical Use Case:
        • For token classification (e.g., masked language modeling).
        • Analyzing the probabilities assigned to specific tokens.
        • Calculating loss during training or evaluation.
           Output: A CausalLMOutputWithPast object that contains:
        • logits: A tensor of shape (batch_size, sequence_length, vocab_size) 
        representing the model’s raw predictions for the next token.
        • past_key_values: Cached past activations for efficient decoding (optional).
        
           Ask GPT: Difference Between model(input_ids) vs. model.generate() 
        in Hugging Face (LLaMA-2-7B)
        """


        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])
    
    for layer, act in acts.items():
        acts[layer] = torch.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts


def main(args):
    datasets = args.datasets
    if datasets == ['all_topic_specific']:
        print("\n=>=> You are triggering 'all_topic_specific'...\ndatasets is: ")
        """
         datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class',
                    'element_symb', 'facts', 'neg_cities', 'neg_sp_en_trans',
                    'neg_inventors', 'neg_animal_class', 'neg_element_symb',
                    'neg_facts', 'cities_conj', 'sp_en_trans_conj',
                    'inventors_conj', 'animal_class_conj', 'element_symb_conj',
                    'facts_conj', 'cities_disj', 'sp_en_trans_disj',
                    'inventors_disj', 'animal_class_disj', 'element_symb_disj',
                    'facts_disj', 'larger_than', 'smaller_than', 'cities_de',
                    'neg_cities_de', 'sp_en_trans_de', 'neg_sp_en_trans_de',
                    'inventors_de', 'neg_inventors_de', 'animal_class_de',
                    'neg_animal_class_de', 'element_symb_de', 'neg_element_symb_de',
                    'facts_de', 'neg_facts_de']
        """
        datasets = ['cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb', 'facts', 
                    'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class',
                    'neg_element_symb', 'neg_facts']
        print(datasets)
        print("\n=>=> Finish triggering 'all_topic_specific'!\n")
    if datasets == ['all']:
        print("\n=>=> You are triggering 'all'...\n")
        datasets = []
        for file_path in glob.glob('datasets/**/*.csv',
                                   recursive=True):
            print("\n=> Reading file: {}".format(file_path))
            dataset_name = (os.path.relpath(file_path, 'datasets')
                            .replace('.csv', ''))
            print("=> Dataset name: {}\n".format(dataset_name))
            datasets.append(dataset_name)
        print("The datasets is:\n{}".format(datasets))
        print("\n=>=> Finish triggering 'all'!\n")

    torch.set_grad_enabled(False)

    print("\n=>=> You are loading the tokenizer and the model...")
    tokenizer, model = load_model(pruning=args.pruning, pruning_ratio=args.pruning_ratio,
                                  model_family=args.model_family,
                                  model_size=args.model_size,
                                  model_type=args.model_type)
    model = model.to(args.device)
    print("\n=>=> Finish loading the tokenizer and the model!")

    for dataset in datasets:
        print("\n=> Loading statements from {}...".format(dataset))
        statements_infor = load_statements(dataset_name=dataset)
        print('\n', '=' * 10, "statements_infor",
              type(statements_infor))
        print(statements_infor)
        print('\n')
        print("\n=> Finish loading statements!")

        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        dataset_save_dir = f"{args.save_dir}/{dataset}"
        print("\n=> dataset save_dir: {}\n".format(dataset_save_dir))
        if not os.path.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)

        for idx in range(0, len(statements_infor), 25):
            print("\n=> Getting acts...")
            acts = get_acts(statements=statements_infor[idx:idx + 25],
                            tokenizer=tokenizer, model=model,
                            layers=layers, device=args.device)
            print('\n', '=' * 10, "acts", type(acts))
            print(acts)
            print('\n')
            print("\n=> Finish getting acts!")
            for layer, act in acts.items():
                torch.save(act, f"{dataset_save_dir}/"
                                f"layer_{layer}_{idx}.pt")



if __name__ == "__main__":
    """
       Read statements from dataset, record activations in 
    given layers, and save to specified files.
    """

    parser = argparse.ArgumentParser(description="Generate activations "
                                                 "for statements in a dataset")
    parser.add_argument("--pruning", type=bool, default=False)
    parser.add_argument("--pruning_ratio", type=float, default=0.7)
    parser.add_argument("--model_family", default="Llama3",
                        choices=["Llama2", "Llama3", "Gemma",
                                 "Gemma2", "Mistral"],
                        help="Model family to use.")
    parser.add_argument("--model_size", default="8B",
                        help="Size of the model to use. "
                             "For Llama2: 7B, 13B, or 70B."
                             "For Llama3: 8B or 70B.")
    parser.add_argument("--model_type", default="base",
                        choices=["base", "chat"],
                        help="Whether to choose base or chat models.")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from. "
                             "Specific: 11 12, or all layers: -1.")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension. "
                             "Options: [neg_cities, cities neg_cities "
                             "sp_en_trans neg_sp_en_trans, "
                             "all_topic_specific, all]")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to.")
    parser.add_argument("--device", default="cuda:3")
    args_infor = parser.parse_args()

    # 'c4_wanda0.5', 'c4_owl0.5', 
    model_type_list = ['c4_wanda_option1_0.5', 'c4_owl_option2_0.5_0.04', 
                       'c4_truthqa_wanda0.5', 'c4_truthqa_owl0.5',
                       'c4_truthqa_wanda_option1_0.5',
                       'c4_truthqa_owl_option2_0.5_0.04']
    model_type_list = ['c4_owl', 'c4_truthqa_owl']
    for args_infor.model_type in model_type_list:
        if args_infor.pruning:
            print("\n=>=> You are triggering the pruning mode...")
            save_dir = (f"{args_infor.output_dir}/{args_infor.model_family}"
                        f"/{args_infor.model_size}/{args_infor.model_type}/")
        else:
            print("\n=>=> You aren't triggering the pruning mode...")
            args_infor.model_type = 'chat'
            save_dir = (f"{args_infor.output_dir}/{args_infor.model_family}"
                        f"/{args_infor.model_size}/{args_infor.model_type}/")
            print("=>=> save_dir: {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, "logs.txt")
        print("=>=> log_file is {}\n".format(log_file))
        sys.stdout = Logger(log_file)
        args_infor.save_dir = save_dir
        main(args=args_infor)
        if args_infor.model_type == 'chat':
            break



    
    
