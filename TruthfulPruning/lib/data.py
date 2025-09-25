# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import json
import torch
import random
import numpy as np
from datasets import load_dataset, Dataset


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    """Wrapper for tokenized input IDs."""
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """Load and process wikitext2 dataset."""
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer, data_seqlen=None):
    """
       Load and process c4 dataset.
    """
    # Load train and validation datasets
    """from datasets import load_dataset"""
    # traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', data_files={'validation':'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    train_file = 'lib/dataset/dataset/c4_en/c4-train.00000-of-01024.json.gz'
    traindata = load_dataset('json',
                             data_files={'train': train_file},
                             split='train')
    pad_token = tokenizer(traindata[0]['text'], return_tensors='pt',
                          padding='max_length', max_length=100000).input_ids[0][0]
    print('\n=>=> pad_token', pad_token)
    print("=> type(traindata) is {}".format(type(traindata))) # <class 'datasets.arrow_dataset.Dataset'>
    print("=> traindata is {}".format(traindata))
    """
    Dataset({
    features: ['text', 'timestamp', 'url'],
    num_rows: 356317
    """
    print("=> len(traindata) is {}".format(len(traindata))) # 356317
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            i_text = traindata[i]['text']
            # print(type(i_text), i_text) # <class 'str'>
            trainenc = tokenizer(i_text, return_tensors='pt')
            """
            A BatchEncoding object containing tokenized data as PyTorch tensors.
            <class 'transformers.tokenization_utils_base.BatchEncoding'> 
            trainenc.input_ids.shape[1] torch.Size([1, 1662])
            {'input_ids': tensor([[31437,  1587,   279,  ...,   520, 65660,  1210]]), The tokenized IDs representing the text.
             'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}, Indicates which tokens are valid (1) and which are padding (0).
            """
            if (data_seqlen is not None
                    and trainenc.input_ids.shape[1] > data_seqlen):
                """ New from OWL!!! 
                - data_seqlen: Number of meaningful tokens in 
                each calibration sample, the remaining portion 
                of context window is filled with padding token.
                """
                break
            if trainenc.input_ids.shape[1] > seqlen:
                """ Same as OWL!!!
                - seqlen: Length of context window in tokens, 
                default as 2048
                   Purpose: Prevent samples that are too short for the model
                by ensuring the tokenized string has more than seqlen tokens. 
                Otherwise, the loop will continue to sample another string.
                """
                break
        
        """
        i: Random starting index within the valid range (0 to total_tokens - seqlen).
        j: The ending index, ensuring a span of exactly seqlen tokens.
        selected_i_to_j_tokens: A slice of trainenc.input_ids containing seqlen tokens.
           Why Not Process the Entire String?
        1. Model Constraints:
        • Most models cannot process sequences longer than max_position_embeddings in 
        one pass. This makes it necessary to split long strings into smaller chunks.
        2. Training Efficiency:
        • Processing the entire string, especially for very long texts, can lead to:
        • GPU memory exhaustion.
        • Inefficient batching during training.
        3. Improved Generalization:
        • By randomly sampling different parts of the string (i:j), the model sees 
        diverse contexts, improving its ability to generalize.
           Purpose:
        • Fixed input length: Always exactly seqlen tokens, regardless of the 
        original text length.
        • Randomization: Different parts of the string are selected for training, introducing 
        variety and preventing the model from overfitting to specific token positions.
        • Compliance with model constraints: It keeps the input length within the bounds 
        the model can handle in a single forward pass.
        """
        if data_seqlen is not None:
            """ New from OWL!!! """
            i = random.randint(0, trainenc.input_ids.shape[1]
                               - data_seqlen - 1)
            j = i + data_seqlen
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
        i_to_j_tokens_input = trainenc.input_ids[:, i:j]
        if data_seqlen is not None:
            """ New from OWL!!! """
            inp = torch.nn.functional.pad(i_to_j_tokens_input,
                                          (seqlen - i_to_j_tokens_input.shape[1], 0),
                                          'constant', pad_token)

        i_to_j_tokens_target = i_to_j_tokens_input.clone()
        i_to_j_tokens_target[:, :-1] = -100
        """
           Purpose: Masking the Target (Labels) by setting all tokens in the tar tensor 
        except for the last one to -100. This operation is significant for training a 
        causal language model. In PyTorch’s loss functions (e.g., CrossEntropyLoss), 
        -100 is often used as a special ignore index. Tokens with the value -100 are 
        ignored when computing the loss, meaning they do not contribute to the 
        model’s gradient updates during training. In this context, the model is being 
        trained to predict the next token at each position in the sequence.
        """
        trainloader.append((i_to_j_tokens_input, i_to_j_tokens_target))
    """
       type(trainloader): <class 'list'>
       len(trainloader): 128 num_samples
       type(trainloader[0]): tuple
       len(trainloader[0]: 2
       trainloader[0][0].size(): torch.Size([1, 2048]), trainloader[0][1].size(): torch.Size([1, 2048])
       trainloader[1][0].size(): torch.Size([1, 2048]), trainloader[1][1].size(): torch.Size([1, 2048])
       trainloader[2][0].size(): torch.Size([1, 2048]), trainloader[2][1].size(): torch.Size([1, 2048])
       trainloader[3][0].size(): torch.Size([1, 2048]), trainloader[3][1].size(): torch.Size([1, 2048])
    """
    
    return trainloader, None, None


def get_pile(nsamples, seed, seqlen, tokenizer):
    print("\n=>=> Getting the pile data...")
    # Load train and validation datasets
    data_path = 'lib/dataset/dataset/pile/00.jsonl.zst'
    traindata = load_dataset("json", data_files=data_path, split="train")
    # traindata = traindata["train"]
    """
     valdata = load_dataset('monology/pile-uncopyrighted',
                           split='train[5000:5100]', verification_mode='no_checks') 
    """
   
    print('=>=> len(traindata) is {}'.format(len(traindata)))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    #valenc = tokenizer(' '.join(valdata['text']), return_tensors='pt')
    #valenc = valenc.input_ids[:, :(256 * seqlen)]
    #valenc = TokenizerWrapper(valenc)
    return trainloader, None, None


def get_oscar(nsamples, seed, seqlen, tokenizer):
    print("\n=>=> Getting the Oscar data...")
    # Load train and validation datasets
    traindata = load_dataset('oscar-corpus/oscar',
                             'unshuffled_deduplicated_en',
                             split='train[:1000]',  # 5000
                             verification_mode='no_checks')
    valdata = load_dataset('oscar-corpus/oscar',
                           'unshuffled_deduplicated_en',
                           split='train[1000:2000]', # 5000:10000
                           verification_mode='no_checks')
    print('\n=>=> len(traindata)', len(traindata))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            print('i=', i)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc



def get_redpajama(nsamples, seed, seqlen, tokenizer):
    print("\n=>=> Getting the RedPajama data...")
    # Load train and validation datasets
    #load_dataset('togethercomputer/RedPajama-Data-V2', 'sample', split='train', verification_mode='no_checks')
    
    traindata = load_dataset('togethercomputer/RedPajama-Data-V2',
                             'sample', split='train[:1000]', languages=["en"], verification_mode='no_checks')
    valdata = load_dataset('togethercomputer/RedPajama-Data-V2',
                           'sample', split='train[1000:2000]', languages=["en"], verification_mode='no_checks')
    
    print("\n=>=> len(traindata): ", len(traindata))
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['raw_content'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['raw_content']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
 

    return trainloader, valenc




def get_gsm8k(nsamples, seed, seqlen, tokenizer, rationale=False,
              padding_side='left', input_format='single',
              num_incontext=None, num_cot_steps=None):
    print("\n=>=> Getting GSM8K...\n")

    def prepare_input(example):
        """
           A helper function to process each dataset example.
           example: <class 'datasets.formatting.formatting.LazyRow'>
        --Example 1:
        -Before:
        {'question': 'Natalia sold clips to 48 of her friends in April, 
        and then she sold half as many clips in May. How many clips did Natalia sell 
        altogether in April and May?', 
        'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n
        #### 72'}
        -After:
        {'question': 'Natalia sold clips to 48 of her friends in April, 
        and then she sold half as many clips in May. How many clips did Natalia sell 
        altogether in April and May?', 
        'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n
        #### 72', 
        'prompt': 'Question: Natalia sold clips to 48 of her friends in April, 
        and then she sold half as many clips in May. How many clips did Natalia sell 
        altogether in April and May?\nAnswer: ', 
        'label': '72', 
        'rationale': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.', 
        'rationale_label': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.. 
        So the answer is 72.', 
        'question_label': 'Question: Natalia sold clips to 48 of her friends in April, 
        and then she sold half as many clips in May. How many clips did Natalia sell 
        altogether in April and May?\nAnswer: 72', 
        'question_rationale_label': 'Question: Natalia sold clips to 48 of her friends 
        in April, and then she sold half as many clips in May. How many clips did Natalia 
        sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips 
        in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and 
        May. So the answer is 72.'}
        """
        print("\n=> Begin processing:\n{}\n".format(example))
        question = example['question']  # Extract the question
        answer = example['answer']      # Extracts the answer

        """
           Splits the answer into:
        • rationale_infor: Reasoning portion before ####.
        • label: The final answer.
        """
        rationale_infor, label = answer.split('####')
        """
           string.strip(): remove leading and trailing characters from a string. 
        By default, it removes whitespace (spaces, tabs, newlines) from both the 
        beginning and the end of the string.
        """
        rationale_infor, label = rationale_infor.strip(), label.strip() 

        example['prompt'] = (f'Question: {question}\n'
                             f'Answer: ')
        example['label'] = label
        example['rationale'] = rationale_infor
        if rationale_infor[-1] != '.':
            rationale_infor = rationale_infor + '.'
        example['rationale_label'] = (f'{rationale_infor} '
                                      f'So the answer is {label}.')
        example['question_label'] = (f'Question: {question}\n'
                                     f'Answer: {label}')
        example['question_rationale_label'] = (f'Question: {question}\n'
                                               f'Answer: {rationale_infor} '
                                               f'So the answer is {label}.')
        print("\n=> The processed example is:\n{}\n".format(example))

        return example

    traindata = load_dataset('gsm8k', 'main', split='train', verification_mode='no_checks')
    traindata_len = len(traindata)
    print("=>=> type(): {}, len() {}, :{}\n".format(type(traindata), len(traindata), traindata))

    print("\n=>=> Begin mapping training data...\n")
    traindata = traindata.map(prepare_input)
    print("\n=>=> Finish mapping training data!\n")
    print("=>=> type(): {}, len() {}, :{}\n".format(type(traindata), len(traindata), traindata))
    """
    traindata: <class 'datasets.arrow_dataset.Dataset'>, 
    Dataset({
    features: ['question', 'answer', 'prompt', 'label', 
    'rationale', 'rationale_label', 'question_label', 
    'question_rationale_label'],
    num_rows: 7473
    })
    """
    
    print("\n=>=> Begin mapping test data...\n")
    testdata = load_dataset('gsm8k', 'main', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)
    print("\n=>=> Finish mapping test data!\n")

    if padding_side == 'left':
        # Extracts the first token from the tokenized input.
        pad_token = tokenizer(traindata[0]['question'],
                              return_tensors='pt',
                              padding='max_length',
                              max_length=seqlen).input_ids[0][0]
        
    elif padding_side == 'right':
        # Extracts the last token from the tokenized input.
        pad_token = tokenizer(traindata[0]['question'],
                              return_tensors='pt',
                              padding='max_length',
                              max_length=seqlen).input_ids[0][-1]
    else:
        assert padding_side in ['left', 'right']
    """ pad_token:
    <class 'torch.Tensor'> tensor(128256)
    """

    random.seed(seed)
    trainloader = []
    if num_incontext is not None:
        num_icl_examples = num_incontext 
        print("\n=>=> Option 1: constrain the number of examples "
              "for ICL as: {}\n".format(num_icl_examples))
    else:
        num_icl_examples = float('inf')
        print("\n=>=> Option 2: Not constrain the number of examples "
              "for ICL until rainenc_concat.input_ids.shape[1] > {}\n".format(seqlen))
    
   
    while len(trainloader) < nsamples: 
        if input_format == 'concat':
            """
            • Handles concatenation of multiple examples.
            • Adds examples until their combined length exceeds seqlen.
            """
            
            idx = 0
            trainenc_concat = None
            while idx < num_icl_examples: # Option 1: constrain the # of examples for ICL if not inf
                idx += 1
                i = random.randint(0, traindata_len - 1)
                if rationale:
                    if idx == 1: # print once
                        print("=> Triggering rationale...")
                    trainenc = tokenizer(traindata[i]['question_rationale_label'],
                                         return_tensors='pt')
                    if num_cot_steps:
                        if idx == 0: # print once
                            print("=> Triggering {} CoT steps...".format(num_cot_steps))
                        curr_num_cot_steps = traindata[i]['rationale_label'].count('. ')
                        """
                        if curr_num_cot_steps != 1:
                            # Might be 1, 2, 3, 4, 5, 6, even 28...
                            print("=> curr_num_cot_steps: {}".format(curr_num_cot_steps))
                        """
                        if num_cot_steps != curr_num_cot_steps:
                            continue
                else:
                    trainenc = tokenizer(traindata[i]['question_label'],
                                         return_tensors='pt')
                
                """
                   Concatenates tokenized examples along the 
                sequence dimension.
                """
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = (
                            torch.concat([trainenc_concat[key],
                                          trainenc[key]], dim=1))
                        
                if trainenc_concat.input_ids.shape[1] > seqlen:
                    """
                    Option 2: Not constrain the number of examples for 
                    incontext learning until rainenc_concat.input_ids.shape[1] > {}
                    """
                    trainenc = trainenc_concat
                    break

            print(f'=> For the sample {len(trainloader) + 1}/{nsamples}, it has {idx} ICL examples')
            
            if num_incontext is not None:
                inp = trainenc_concat.input_ids
                print("=>=> num_incontext is not None, need padding {}!".format(padding_side))
                """
                   pad is a PyTorch function that pads a tensor 
                on its dimensions with specified values.
                (seqlen - inp.shape[1], 0): pad the sequence with 256 - 100 = 156 zeros on the left.
                (0, seqlen - inp.shape[1]): pad the sequence with 256 - 100 = 156 zeros on the right.
                'constant' means the padding will use a fixed value (pad_token)
                """
                if padding_side == 'left':
                    inp = torch.nn.functional.pad(inp,
                                                  (seqlen - inp.shape[1], 0),
                                                  'constant', pad_token)
                elif padding_side == 'right':
                    inp = torch.nn.functional.pad(inp,
                                                  (0, seqlen - inp.shape[1]),
                                                  'constant', pad_token)
                else:
                    assert padding_side in ['left', 'right']
            else:
                print("=>=> num_incontext is None, no need padding!")
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
            
            tar = inp.clone() # The tokenized input.
            tar[:, :-1] = -100 # The label for training
            trainloader.append((inp, tar))

        elif input_format == 'single':
            i = random.randint(0, traindata_len - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['question_rationale_label'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['question_label'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

        elif input_format == 'zero':
            i = random.randint(0, traindata_len - 1)
            trainenc = tokenizer(traindata[i]['prompt'],
                                 padding='max_length',
                                 max_length=seqlen,
                                 return_tensors='pt')
            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))
        else:
            print("=>=> Input format {} isn't supported!".format(input_format))
            exit(-1)

    test_prompt = testdata[:nsamples]['prompt']
    test_answer = testdata[:nsamples]['label']

    print("\n=>=> Finish getting GSM8K!\n")

    return trainloader, (test_prompt, test_answer), None


def get_truthIsUniversal(nsamples, seed, seqlen, tokenizer, statement_num_value, statement_only,
                         padding_side='left', input_format='single'):
    print("\n=>=> Getting data from the Truth-is-Universal project...\n")
    with open("./lib/dataset/dataset/truth_is_universal/all_datasets.json", 'r') as json_file:
        datasets_list = json.load(json_file)
    print(len(datasets_list))  # 9214
    print(datasets_list[0])
    """
    {'statement_only': 'The city of Krasnodar is not in Russia.', 
    'statement_with_label': 'The city of Krasnodar is not in Russia, 
    which is false and the correct answer is indeed Russia.'}
    """
    # Create a Dataset from the list of dictionaries
    traindata = (
        Dataset.from_dict({
            'statement_only': [item['statement_only'] for item in datasets_list],
            'statement_with_label': [item['statement_with_label'] for item in datasets_list]
        })
    )
    traindata_len = len(traindata)
    print("=>=> type(): {}, traindata_len: {}, "
          "traindata:{}\n".format(type(traindata),
                                  traindata_len,
                                  traindata))
    """
    Dataset({
    features: ['statement_only', 'statement_with_label'],
    num_rows: 9214
    })
    """
    pad_token = None
    if padding_side == 'left':
        # Extracts the first token from the tokenized input.
        pad_token1 = tokenizer(traindata[0]['statement_only'],
                               return_tensors='pt',
                               padding='max_length',
                               max_length=seqlen).input_ids[0][0]
        pad_token2 = tokenizer(traindata[0]['statement_with_label'],
                               return_tensors='pt',
                               padding='max_length',
                               max_length=seqlen).input_ids[0][0]
        assert pad_token1 == pad_token2
        pad_token = pad_token1

    elif padding_side == 'right':
        # Extracts the last token from the tokenized input.
        pad_token1 = tokenizer(traindata[0]['statement_only'],
                               return_tensors='pt',
                               padding='max_length',
                               max_length=seqlen).input_ids[0][-1]
        pad_token2 = tokenizer(traindata[0]['statement_with_label'],
                               return_tensors='pt',
                               padding='max_length',
                               max_length=seqlen).input_ids[0][-1]
        assert pad_token1 == pad_token2
        pad_token = pad_token1
    else:
        assert padding_side in ['left', 'right']
    assert pad_token is not None
    print("\n=>=> pad_token is {}\n".format(pad_token))
    """ pad_token:
        <class 'torch.Tensor'> tensor(128256)
    """

    random.seed(seed)
    trainloader = []
    if statement_num_value is not None:
        statement_num = statement_num_value
        print("\n=>=> Option 1: constrain the number of statements "
              "as: {}\n".format(statement_num))
    else:
        statement_num = float('inf')
        print("\n=>=> Option 2: Not constrain the number of statements "
              "until rainenc_concat.input_ids.shape[1] > {}\n".format(seqlen))

    print("\n=>=> input_format is {}\n".format(input_format))
    total_selected_num = 0
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            """
            • Handles concatenation of multiple examples.
            • Adds examples until their combined length exceeds seqlen.
            """
            idx = 0
            trainenc_concat = None
            while idx < statement_num:  # Option 1: constrain the # of statements if not inf
                idx += 1
                i = random.randint(0, traindata_len - 1)
                if statement_only:
                    trainenc = tokenizer(traindata[i]['statement_only'],
                                         return_tensors='pt')
                else:
                    trainenc = tokenizer(traindata[i]['statement_with_label'],
                                         return_tensors='pt')
                """
                   Concatenates tokenized examples along the 
                sequence dimension.
                """
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = (
                            torch.concat([trainenc_concat[key],
                                          trainenc[key]], dim=1))
                if trainenc_concat.input_ids.shape[1] > seqlen:
                    """
                    Option 2: Not constrain the number of examples for 
                    incontext learning until rainenc_concat.input_ids.shape[1] > {}
                    """
                    trainenc = trainenc_concat
                    break
            if statement_only:
                print(f'=> For the sample {len(trainloader) + 1}/{nsamples}, '
                      f'it has {idx} statements-only examples.')
            else:
                print(f'=> For the sample {len(trainloader) + 1}/{nsamples}, '
                      f'it has {idx} statements-with-label examples.')
            total_selected_num += idx


            if statement_num_value is not None:
                inp = trainenc_concat.input_ids
                print("=>=> statement_num_value is not None, need padding {}!".format(padding_side))
                """
                   pad is a PyTorch function that pads a tensor 
                on its dimensions with specified values.
                (seqlen - inp.shape[1], 0): pad the sequence with 256 - 100 = 156 zeros on the left.
                (0, seqlen - inp.shape[1]): pad the sequence with 256 - 100 = 156 zeros on the right.
                'constant' means the padding will use a fixed value (pad_token)
                """
                if padding_side == 'left':
                    inp = torch.nn.functional.pad(inp,
                                                  (seqlen - inp.shape[1], 0),
                                                  'constant', pad_token)
                elif padding_side == 'right':
                    inp = torch.nn.functional.pad(inp,
                                                  (0, seqlen - inp.shape[1]),
                                                  'constant', pad_token)
                else:
                    assert padding_side in ['left', 'right']
            else:
                print("=>=> statement_num_value is None, no need padding!")
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]

            tar = inp.clone()  # The tokenized input.
            tar[:, :-1] = -100  # The label for training
            trainloader.append((inp, tar))
            

        elif input_format == 'single':
            i = random.randint(0, traindata_len - 1)
            if statement_only:
                trainenc = tokenizer(traindata[i]['statement_only'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['statement_with_label'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

        else:
            print("=>=> Input format {} isn't supported!".format(input_format))
            exit(-1)

    
    print(f"\n=>=> Finish getting data from the Truth-is-Universal project, where {total_selected_num} are selected from {traindata_len}!\n")
    

    return trainloader, None, None


def get_enriched_truth_qa(nsamples, seed, seqlen, tokenizer, statement_num_value, statement_only,
                          padding_side='left', input_format='single'):
    print("\n=>=> Getting data from the enriched truthfulQA dataset...\n")
    file_path = "./lib/dataset/dataset/enriched_truthfulqa/enrichedTruthfulQA.txt"
    with open(file_path, 'r') as file:
        datasets_list = []  # List to store strings
        current_string = ""  # Temp string to hold concatenated lines

        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:  # If the line is not empty
                current_string += line + " "  # Add line and a space
            else:  # Encounter a blank line
                if current_string:  # If there's a string to save
                    datasets_list.append(current_string.strip())  # Save it, remove trailing space
                current_string = ""  # Reset for the next string

        # Add the last string if not already added
        if current_string:
            datasets_list.append(current_string.strip())


    ## Create a Dataset from the list of dictionaries
    traindata = (
        Dataset.from_dict({
            'statement_with_clarification': [item for item in datasets_list]
        })
    )
    traindata_len = len(traindata)
    print("=>=> type(): {}, traindata_len: {}, "
          "traindata:{}\n".format(type(traindata),
                                  traindata_len,
                                  traindata))
    """
    Dataset({
    features: ['statement_with_clarification'],
    num_rows: 
    })
    """
    pad_token = None
    if padding_side == 'left':
        # Extracts the first token from the tokenized input.
        pad_token = tokenizer(traindata[0]['statement_with_clarification'],
                              return_tensors='pt',
                              padding='max_length',
                              max_length=seqlen).input_ids[0][0]

    elif padding_side == 'right':
        # Extracts the last token from the tokenized input.
        pad_token = tokenizer(traindata[0]['statement_with_clarification'],
                              return_tensors='pt',
                              padding='max_length',
                              max_length=seqlen).input_ids[0][-1]
    else:
        assert padding_side in ['left', 'right']
    assert pad_token is not None
    print("\n=>=> pad_token is {}\n".format(pad_token))
    """ pad_token:
        <class 'torch.Tensor'> tensor(128256)
    """

    random.seed(seed)
    trainloader = []
    if statement_num_value is not None:
        statement_num = statement_num_value
        print("\n=>=> Option 1: constrain the number of statements "
              "as: {}\n".format(statement_num))
    else:
        statement_num = float('inf')
        print("\n=>=> Option 2: Not constrain the number of statements "
              "until rainenc_concat.input_ids.shape[1] > {}\n".format(seqlen))

    print("\n=>=> input_format is {}\n".format(input_format))
    total_selected_num = 0
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            """
            • Handles concatenation of multiple examples.
            • Adds examples until their combined length exceeds seqlen.
            """
            idx = 0
            trainenc_concat = None
            while idx < statement_num:  # Option 1: constrain the # of statements if not inf
                idx += 1
                i = random.randint(0, traindata_len - 1)
                if statement_only:
                    trainenc = tokenizer(traindata[i]['statement_without_clarification'],
                                         return_tensors='pt')
                else:
                    trainenc = tokenizer(traindata[i]['statement_with_clarification'],
                                         return_tensors='pt')
                """
                   Concatenates tokenized examples along the 
                sequence dimension.
                """
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = (
                            torch.concat([trainenc_concat[key],
                                          trainenc[key]], dim=1))
                if trainenc_concat.input_ids.shape[1] > seqlen:
                    """
                    Option 2: Not constrain the number of examples for 
                    incontext learning until rainenc_concat.input_ids.shape[1] > {}
                    """
                    trainenc = trainenc_concat
                    break
            if statement_only:
                print(f'=> For the sample {len(trainloader) + 1}/{nsamples}, '
                      f'it has {idx} statements-only examples.')
            else:
                print(f'=> For the sample {len(trainloader) + 1}/{nsamples}, '
                      f'it has {idx} statements-with-label examples.')
            total_selected_num += idx

            if statement_num_value is not None:
                inp = trainenc_concat.input_ids
                print("=>=> statement_num_value is not None, need padding {}!".format(padding_side))
                """
                   pad is a PyTorch function that pads a tensor 
                on its dimensions with specified values.
                (seqlen - inp.shape[1], 0): pad the sequence with 256 - 100 = 156 zeros on the left.
                (0, seqlen - inp.shape[1]): pad the sequence with 256 - 100 = 156 zeros on the right.
                'constant' means the padding will use a fixed value (pad_token)
                """
                if padding_side == 'left':
                    inp = torch.nn.functional.pad(inp,
                                                  (seqlen - inp.shape[1], 0),
                                                  'constant', pad_token)
                elif padding_side == 'right':
                    inp = torch.nn.functional.pad(inp,
                                                  (0, seqlen - inp.shape[1]),
                                                  'constant', pad_token)
                else:
                    assert padding_side in ['left', 'right']
            else:
                print("=>=> statement_num_value is None, no need padding!")
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]

            tar = inp.clone()  # The tokenized input.
            tar[:, :-1] = -100  # The label for training
            trainloader.append((inp, tar))


        elif input_format == 'single':
            i = random.randint(0, traindata_len - 1)
            if statement_only:
                trainenc = tokenizer(traindata[i]['statement_only'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['statement_with_clarification'],
                                     padding='max_length',
                                     max_length=seqlen,
                                     return_tensors='pt')
            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

        else:
            print("=>=> Input format {} isn't supported!".format(input_format))
            exit(-1)

    print(
        f"\n=>=> Finish getting data from the Truth-is-Universal project, "
        f"where {total_selected_num} are selected from {traindata_len}!\n")

    return trainloader, None, None



def get_truth_qa(nsamples, seed, seqlen, tokenizer):
    """
       Load and process the TruthQA dataset.
    """
    # print(os.getcwd())
    train_file = './lib/dataset/dataset/truth_qa/TruthfulQA.json'
    with open(train_file, 'r') as json_file:
        truthqa_list = json.load(json_file)
    print(len(truthqa_list))
    print(truthqa_list[0])
    # Create a Dataset from the list of dictionaries
    """from datasets import Dataset"""
    traindata = (
        Dataset.from_dict(
            {'text': [item['text'] for item in truthqa_list],
             'timestamp': [item['timestamp'] for item in truthqa_list],
             'url': [item['url'] for item in truthqa_list]
             }
        )
    )

    print("=> type(traindata) is {}".format(type(traindata))) # <class 'datasets.arrow_dataset.Dataset'>
    print("=> traindata is {}".format(traindata))
    """
    Dataset({
    features: ['text', 'timestamp', 'url'],
    num_rows: 817
    """
    print("=> len(traindata) is {}\n".format(len(traindata))) # 817

    trainloader = []
    token_length = []
    for i in range(len(traindata)):
        i_text = traindata[i]['text']
        print(type(i_text), i_text) # <class 'str'>
        
        tokenizer.padding_side = "right"  # Padding added to the right
        tokenizer.pad_token = tokenizer.eos_token # Set the existing eos_token as the pad_token
        trainenc = tokenizer(i_text, return_tensors='pt')
        # trainenc = tokenizer(i_text, return_tensors='pt', padding=True, max_length=256, truncation=True)
        print(type(trainenc), trainenc.input_ids.shape)
        print(type(trainenc), trainenc.attention_mask.shape)
        
        
        current_token_length = trainenc.input_ids.shape[1]
        token_length.append(current_token_length)
        last_idx = current_token_length - 1

        # Calculate the number of padding tokens needed
        padding_length = seqlen - current_token_length
        assert current_token_length <= seqlen

        # Pad 'input_ids' with 0
        trainenc["input_ids"] = torch.cat(
            [trainenc["input_ids"], torch.zeros((1, padding_length), dtype=torch.long)], dim=1
        )
        
        # Pad 'attention_mask' with 0
        trainenc["attention_mask"] = torch.cat(
            [trainenc["attention_mask"], torch.zeros((1, padding_length), dtype=torch.long)], dim=1
        )
        
        
        print(type(trainenc), trainenc.input_ids.shape)
        print(type(trainenc), trainenc.attention_mask.shape)
        
        tokens_input = trainenc.input_ids
        tokens_target = tokens_input.clone()
        tokens_target[:, :last_idx] = -100

        """
           Purpose: Masking the Target (Labels) by setting all tokens in 
        the tar tensor except for the last one to -100. This operation is 
        significant for training a causal language model. In PyTorch’s loss 
        functions (e.g., CrossEntropyLoss), -100 is often used as a special 
        ignore index. Tokens with the value -100 are ignored when computing 
        the loss, meaning they do not contribute to the model’s gradient 
        updates during training. In this context, the model is being 
        trained to predict the next token at each position in the sequence.
        """
        trainloader.append((tokens_input, tokens_target))
        
    print(f"=>=> max token length is {max(token_length)}, "
          f"min token length is {min(token_length)}, "
          f"total token length is {sum(token_length)}\n")
 

    return trainloader, None, None


def get_svamp_rationale(nsamples, seed, seqlen, tokenizer,
                        input_format='single', padding_side='left',
                        mode='llama', verbose=False):
    print('get_svamp_rationale')
    dataset_loader_svamp = SVAMPDatasetLoader()
    datasets_svamp = dataset_loader_svamp.load_from_json()

    llm_rationales_svamp = {}
    llm_labels_svamp = {}
    question_rationale_label = {}
    prompts = {}
    labels = {}
    for split in ['train', 'test']:
        llm_rationales_svamp[split], llm_labels_svamp[split] = dataset_loader_svamp.load_llm_preds(split=split)

        question_rationale_label[split] = np.char.array(datasets_svamp[split]['input']) + '\nAnswer: ' + np.char.array(
            llm_rationales_svamp[split]) + '. So the answer is ' + datasets_svamp[split]['label']
        prompts[split] = 'Question: ' + np.char.array(datasets_svamp[split]['input']) + '.\nAnswer: '
        labels[split] = np.char.array(datasets_svamp[split]['label'])

    traindata = Dataset.from_dict({'question_rationale_label': question_rationale_label['train'],
                                   'prompt': prompts['train'],
                                   'label': labels['train']})
    testdata = Dataset.from_dict({'question_rationale_label': question_rationale_label['test'],
                                   'prompt': prompts['test'],
                                   'label': labels['test']})
    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['question_rationale_label'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['question_rationale_label'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][-1]

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['question_rationale_label'],
                                     return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = (
                            torch.concat([trainenc_concat[key],
                                          trainenc[key]], dim=1)
                        )

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['question_rationale_label'],
                                 return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp,
                                                          (seqlen - inp.shape[1], 0),
                                                          'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp,
                                                          (0, seqlen - inp.shape[1]),
                                                          'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['question_rationale_label'],
                                 padding='max_length', max_length=seqlen,
                                 return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:nsamples]['prompt']
    test_answer = testdata[:nsamples]['label']

    return trainloader, (test_prompt, test_answer), pad_token


def get_svamp_individual(tokenizer, split='train'):
    def prepare_input(example):
        question = example['Body'] + ' ' + example['Question']
        answer = example['Equation']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example
    
    data = load_dataset('ChilleD/SVAMP', split=split, verification_mode='no_checks')
    data = data.map(prepare_input)

    return data


def get_svamp(nsamples, seed, seqlen, tokenizer, rationale=False,
              input_format='concat', padding_side='left',
              mode='llama', verbose=False):
    if rationale:
        get_svamp_rationale(nsamples, seed, seqlen, tokenizer,
                            input_format=input_format, padding_side=padding_side,
                            mode=mode, verbose=verbose)
    def prepare_input(example):
        question = example['Body'] + ' ' + example['Question']
        answer = example['Equation']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('ChilleD/SVAMP',
                             split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('ChilleD/SVAMP',
                            split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key],
                                                             trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp,
                                                          (seqlen - inp.shape[1], 0),
                                                          'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp,
                                                          (0, seqlen - inp.shape[1]),
                                                          'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'],
                                 padding='max_length', max_length=seqlen,
                                 return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']
   
    return trainloader, (test_prompt, test_answer), pad_token


def get_mawps(nsamples, seed, seqlen, tokenizer, rationale=False,
              input_format='concat', padding_side='left',
              mode='llama', verbose=False):
    def prepare_input(example):
        question = example['question']
        answer = example['expression']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('MU-NLPC/Calc-mawps', split='train',
                             verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('MU-NLPC/Calc-mawps', split='test',
                            verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key],
                                                             trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp,
                                                          (seqlen - inp.shape[1], 0),
                                                          'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp,
                                                          (0, seqlen - inp.shape[1]),
                                                          'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length',
                                 max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token


def get_commonsense_qa(nsamples, seed, seqlen, tokenizer,
                       rationale=False, input_format='concat',
                       padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        question = example['question']
        choices = example['choices']
        label = example['answerKey']
        choices_formatted = ''
        for idx, letter in enumerate(choices['label']):
            choices_formatted += f'{letter}. {choices["text"][idx]}'
            if idx < len(choices['label']) - 1:
                choices_formatted += '\n'

        example['prompt'] = f'Question: {question}\nChoices:\n{choices_formatted}\nAnswer: '
        example['answer'] = label
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('commonsense_qa', split='train',
                             verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('commonsense_qa', split='validation',
                            verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt',
                              padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt',
                              padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'],
                                     return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key],
                                                             trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp,
                                                          (seqlen - inp.shape[1], 0),
                                                          'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp,
                                                          (0, seqlen - inp.shape[1]),
                                                          'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'],
                                 padding='max_length', max_length=seqlen,
                                 return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token


def get_winogrande(nsamples, seed, seqlen, tokenizer, rationale=False,
                   input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = example['answer']
        options_formatted = f'1. {option1}\n2. {option2}'

        example['prompt'] = f'Sentence: {sentence}\nOptions:\n{options_formatted}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('winogrande', 'winogrande_m',
                             split='train', verification_mode='no_checks', trust_remote_code=True)
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('winogrande', 'winogrande_m',
                            split='validation', verification_mode='no_checks', trust_remote_code=True)
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'],
                              return_tensors='pt', padding='max_length',
                              max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key],
                                                             trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp,
                                                          (seqlen - inp.shape[1], 0),
                                                          'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp,
                                                          (0, seqlen - inp.shape[1]),
                                                          'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'],
                                 padding='max_length', max_length=seqlen,
                                 return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token



def get_loaders(args, data_name, nsamples=128, seqlen=2048,
                tokenizer=None, data_seqlen=None):
    """
       Select the appropriate loader based on dataset names.
    """
    if 'wikitext2' in data_name:
        return get_wikitext2(nsamples=nsamples, seed=args.seed, seqlen=seqlen, tokenizer=tokenizer)
    elif "c4" in data_name:
        return get_c4(nsamples=nsamples, seed=args.seed,
                      seqlen=seqlen, tokenizer=tokenizer,
                      data_seqlen=data_seqlen)
    elif "pile" in data_name:
        return get_pile(nsamples=nsamples, seed=args.seed,
                        seqlen=seqlen, tokenizer=tokenizer)
    elif "oscar" in data_name:
        return get_oscar(nsamples, seed=args.seed, 
                         seqlen=seqlen, tokenizer=tokenizer)
    elif "redpajama" in data_name:
        return get_redpajama(nsamples, seed=args.seed, 
                             seqlen=seqlen, tokenizer=tokenizer)

    elif 'gsm8k' in data_name:
        return get_gsm8k(nsamples=nsamples, seed=args.seed,
                         seqlen=seqlen, tokenizer=tokenizer,
                         input_format=args.input_format,
                         rationale=args.rationale,
                         padding_side=args.padding_side,
                         num_incontext=args.num_incontext,
                         num_cot_steps=args.num_cot_steps)
    elif 'svamp' in data_name:
        return get_svamp(nsamples, args.seed, seqlen, tokenizer,
                         input_format=args.input_format,
                         rationale=args.rationale,
                         padding_side=args.padding_side, mode='llama',
                         verbose=True)
    elif 'mawps' in data_name:
        return get_mawps(nsamples, args.seed, seqlen, tokenizer,
                         input_format=args.input_format,
                         rationale=args.rationale,
                         padding_side=args.padding_side, mode='llama',
                         verbose=True)
    elif 'commonsense_qa' in data_name:
        return get_commonsense_qa(nsamples, args.seed, seqlen, tokenizer,
                                  input_format=args.input_format,
                                  rationale=args.rationale,
                                  padding_side=args.padding_side,
                                  mode='llama', verbose=True)
    elif 'winogrande' in data_name:
        return get_winogrande(nsamples, args.seed, seqlen, tokenizer,
                              input_format=args.input_format,
                              rationale=args.rationale,
                              padding_side=args.padding_side,
                              mode='llama', verbose=True)
    elif 'truth_is_universal' in data_name:
        return get_truthIsUniversal(nsamples=nsamples, seed=args.seed, 
                                    seqlen=seqlen, tokenizer=tokenizer, 
                                    statement_num_value=None, statement_only=False,
                                    padding_side='left', input_format='concat')
    elif 'enriched_truth_qa' in data_name:
        return get_enriched_truth_qa(nsamples=nsamples, seed=args.seed,
                                     seqlen=seqlen, tokenizer=tokenizer,
                                     statement_num_value=None, statement_only=False,
                                     padding_side='left', input_format='concat')

    elif "truth_qa" in data_name:
        return get_truth_qa(nsamples=nsamples, seed=args.seed,
                            seqlen=seqlen, tokenizer=tokenizer)
    
    else:
        print(f"=>=> The dataset {data_name} is unrecognizable! "
              f"Please refer to the CalibratedPruning project for "
              f"the codes to load datasets")
        exit(-1)
