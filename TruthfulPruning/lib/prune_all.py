import copy
import torch
import numpy as np
import torch.nn as nn 

from .layerwrapper import WrappedGPT
from .data import get_loaders


def prepare_calibration_input(args, model, dataloader, device, pad_token,
                              mode='llama', prepend_calibration=None):
    print('\n=>=> Running prepare_calibration_input')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if mode == 'gpt2':
        layers = model.transformer.h
    elif mode == 'llama':
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if (hasattr(model, "hf_device_map") and "model.embed_tokens"
            in model.hf_device_map):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    print("=>=> prepare_calibration_input model.seqlen", model.seqlen)
    inps = torch.zeros((args.nsamples, model.seqlen,
                        model.config.hidden_size),
                       dtype=dtype, device=device)
    inps.requires_grad = False
    print("=>=> prepare_calibration_input inps.size()", inps.size())

    attention_masks = torch.zeros((args.nsamples, 1, 1,
                                   model.seqlen, model.seqlen),
                                  device=device)
    attention_masks.requires_grad = False
    cache = {"i": 0, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if prepend_calibration is not None:
                #print("forward inp.size()", inp.size())
                #print("forward prepend_calibration.size()", prepend_calibration.size())
                inp[:, :prepend_calibration.size()[1], :] = (
                    prepend_calibration)

            inps[cache['i']] = inp
            if 'attention_mask' in kwargs:
                """
                   There are nsamples masks.
                """
                attention_masks[cache['i']] = kwargs['attention_mask']

            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
                #print("position_ids", kwargs["position_ids"])
                #print("position_ids.size()", kwargs["position_ids"].size())

            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i, batch in enumerate(dataloader):
        inp = batch[0].to(device)
        if prepend_calibration is not None:
            inp = torch.cat((torch.ones(1,
                                        prepend_calibration.size()[1])
                             .to(device), inp), dim=1).type(
                torch.LongTensor
            )

        #print("inp.size()", inp.size())
        try:
            #print('pad_token', pad_token)
            attention_mask = (torch.tensor([
                [0 if inp[0][j] == pad_token else 1
                 for j in range(inp.size()[1])]]).
                              to(device).type(torch.LongTensor))
            #print('attention_mask', attention_mask)
            #print('inp', inp)
            model(inp.to(device), attention_mask=attention_mask.to(device))
            #model(inp.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    #print("model.seqlen", model.seqlen)
    #print("default_attention_mask.size()", default_attention_mask.size())
    return inps, outs, attention_masks, position_ids


def find_layers(module, layers=[nn.Linear], name=''):
    """
       A recursive function that searches through a
    PyTorch model (or module) to find specific types of layers
    (such as nn.Linear), and it collects these layers in a dictionary.
    The function is recursive because it calls itself to search
    within the child modules of the given module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.
    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child, layers=layers,
                        name=name + '.' + name1 if name != '' else name1
                        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def check_sparsity_mask(mask):
    W = mask
    count = 0 
    total_params = 0
    count += (W != 0).sum().item()
    total_params += W.numel()
    print(f" density {float(count)/total_params:.6f}")


def check_outlier(mask, threshold):
    W = mask
    count = 0 
    total_params = 0
    max_shred = torch.max(W) * threshold
    count += (W > max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio = float(count) / total_params * 100
    
    return outlier_ratio


def check_outlier_mean(mask, threshold):
    """
       The “outlier ratio” of A by identifying
    elements whose magnitude is M times greater
    than the averaged value in each layer.
    """
    W = mask
    count = 0 
    total_params = 0
    max_shred = torch.mean(W) * threshold
    count += (W > max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio = float(count) / total_params * 100
    
    return outlier_ratio


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1,
                         index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()

    return W_mask, cur_sparsity


def get_truthfulness_ratio(args, layer_num, scaling_factor=0.04):
    pivot_idx = None
    cities_var_ratio, neg_cities_var_ratio = None, None
    sp_en_trans_var_ratio, neg_sp_en_trans_var_ratio = None, None
    if "Llama-3-8B-Instruct" in args.model:
        print("=>=> You are getting the truthfulness ratio of Llama-3-8B-Instruct...")
        cities_var_ratio = np.array([3.39071090e-04, 1.04245816e-03, 2.39174780e-03, 6.37486248e-03,
                                     3.81021744e-02, 8.09538393e-02, 1.32550373e-01, 2.47270862e-01,
                                     3.81992400e-01, 5.26349089e-01, 6.45404196e-01, 7.53192202e-01,
                                     7.32511960e-01, 6.94884655e-01, 6.29495111e-01, 6.29789162e-01,
                                     5.79958105e-01, 5.05475324e-01, 4.78962535e-01, 4.64287028e-01,
                                     4.53694642e-01, 4.26050788e-01, 4.09579671e-01, 3.94378530e-01,
                                     3.39963516e-01, 3.15667607e-01, 2.97716687e-01, 2.58198872e-01,
                                     2.57569347e-01, 2.37331753e-01, 1.95439694e-01, 1.61557409e-01])
        neg_cities_var_ratio = np.array([2.75635858e-04, 1.04091549e-03, 2.34982485e-03, 6.99114158e-03,
                                         4.09611215e-02, 8.77295960e-02, 1.22624907e-01, 1.88092399e-01,
                                         3.41688715e-01, 5.12884356e-01, 6.27721497e-01, 6.55794854e-01,
                                         1.01014698e+00, 8.92597053e-01, 7.48149017e-01, 6.31548774e-01,
                                         5.36927071e-01, 4.01787252e-01, 3.04532688e-01, 2.92508502e-01,
                                         2.59050552e-01, 2.27964383e-01, 2.02185914e-01, 1.68580469e-01,
                                         9.77188368e-02, 9.09651150e-02, 7.93741087e-02, 6.23002706e-02,
                                         6.09152265e-02, 5.66922517e-02, 4.91064340e-02, 3.92692501e-02])
        sp_en_trans_var_ratio = np.array([0.00650565, 0.00233356, 0.00296728, 0.0100899, 0.01508244, 0.03669597,
                                          0.07735114, 0.12632234, 0.19591362, 0.24608271, 0.31299248, 0.3722659,
                                          0.38678397, 0.32486482, 0.27961429, 0.23536007, 0.21875916, 0.19583675,
                                          0.18217044, 0.17587753, 0.16802919, 0.14759805, 0.13368673, 0.12258105,
                                          0.11283833, 0.10460265, 0.0997852, 0.09395919, 0.08996084, 0.08321178,
                                          0.08815841, 0.11194301])
        neg_sp_en_trans_var_ratio = np.array([0.00566225, 0.00234005, 0.00331782, 0.0127961, 0.01657107, 0.03848792,
                                              0.09450669, 0.16633245, 0.26102221, 0.30004234, 0.3645685, 0.45129881,
                                              0.5066566, 0.47487872, 0.43538618, 0.38333991, 0.38493038, 0.38590715,
                                              0.35965402, 0.34839013, 0.34806121, 0.30753975, 0.26950751, 0.25464624,
                                              0.21906972, 0.19438002, 0.17653393, 0.15905798, 0.15261992, 0.14192056,
                                              0.12835094, 0.11994015])
        pivot_idx = 10
    elif "Llama-2-13b-chat" in args.model:
        cities_var_ratio = np.array([1.34069618e-04, 1.01290691e-04, 1.01395135e-04, 9.11259914e-04,
                                     1.29223651e-02, 3.33032686e-02, 3.47080900e-02, 9.35025613e-02,
                                     1.53762834e-01, 3.95125587e-01, 4.83480971e-01, 6.14879019e-01,
                                     7.18658946e-01, 7.63396798e-01, 7.58919909e-01, 6.96811567e-01,
                                     5.97693556e-01, 5.67152247e-01, 5.41872485e-01, 5.07069382e-01,
                                     4.79610024e-01, 4.26690772e-01, 4.09675636e-01, 3.78543970e-01,
                                     3.60294434e-01, 3.54320887e-01, 3.27635582e-01, 3.26762833e-01,
                                     3.05676849e-01, 2.96628321e-01, 2.86488970e-01, 2.80215814e-01,
                                     2.72507135e-01, 2.65403988e-01, 2.66172511e-01, 2.59197293e-01,
                                     2.47589890e-01, 2.37944439e-01, 2.24037604e-01, 1.95875832e-01])
        neg_cities_var_ratio = np.array([2.43484155e-04, 1.58490869e-04, 1.61418970e-04, 9.44208714e-04,
                                         1.36771279e-02, 3.52960413e-02, 3.62310257e-02, 9.41538934e-02,
                                         1.80279431e-01, 3.05234380e-01, 3.07087058e-01, 4.25119273e-01,
                                         4.38110662e-01, 5.13082457e-01, 5.14731941e-01, 5.36897282e-01,
                                         5.13918546e-01, 4.64267745e-01, 4.57722780e-01, 4.26827796e-01,
                                         4.04295825e-01, 3.34723812e-01, 3.14398389e-01, 2.73027010e-01,
                                         2.53230728e-01, 2.43712560e-01, 2.13677067e-01, 1.99056167e-01,
                                         1.77085069e-01, 1.68766710e-01, 1.61668030e-01, 1.55755802e-01,
                                         1.45841118e-01, 1.45062522e-01, 1.48570915e-01, 1.46010050e-01,
                                         1.38354139e-01, 1.30462898e-01, 1.26544759e-01, 1.16083349e-01])
        sp_en_trans_var_ratio = np.array([0.00152078, 0.00178027, 0.00268346, 0.00278452, 0.00456635, 0.0293738,
                                          0.03840297, 0.09246504, 0.1213476, 0.19529955, 0.22897366, 0.29735841,
                                          0.36551053, 0.44216985, 0.44482454, 0.42751286, 0.41531314, 0.4049494,
                                          0.39049526, 0.37933314, 0.365285, 0.342401, 0.35386232, 0.34439429,
                                          0.33166123, 0.322755, 0.29908096, 0.28927534, 0.27582071, 0.27263517,
                                          0.26554488, 0.26216983, 0.26108605, 0.25580946, 0.25350143, 0.24853751,
                                          0.23717513, 0.22427546, 0.221084, 0.26509958])
        neg_sp_en_trans_var_ratio = np.array([0.0015318, 0.00171704, 0.0027661, 0.00285272, 0.00480887, 0.03423065,
                                              0.04249519, 0.09435088, 0.14787492, 0.2047889, 0.24363107, 0.36754176,
                                              0.44787096, 0.52434068, 0.55734157, 0.54540121, 0.51014682, 0.4868963,
                                              0.47782186, 0.46865264, 0.46173493, 0.42564806, 0.42723793, 0.41091905,
                                              0.39981747, 0.39087626, 0.36620457, 0.35638307, 0.32838264, 0.32262172,
                                              0.31403203, 0.30927054, 0.29992871, 0.29715262, 0.28777251, 0.28127836,
                                              0.26905906, 0.25110724, 0.23763362, 0.20298873])
        pivot_idx = 12
    elif "Mistral-7B-Instruct-v0.3" in args.model:
        cities_var_ratio = np.array([0.00130565, 0.00118005, 0.00120108, 0.00354514, 0.00917601, 0.01735215,
                                     0.03165967, 0.05182141, 0.08941566, 0.13172074, 0.261069, 0.61449051,
                                     0.72145414, 0.85533088, 0.7929497, 0.7869248, 0.76342521, 0.73111682,
                                     0.65978662, 0.56190432, 0.49925527, 0.41522812, 0.37704258, 0.37068638,
                                     0.35657639, 0.35290686, 0.34233666, 0.32951957, 0.30948409, 0.28826947,
                                     0.27430511, 0.26027087])
        neg_cities_var_ratio = np.array([0.00132415, 0.00119984, 0.00118895, 0.00435284, 0.00952015, 0.01906506,
                                         0.03648416, 0.08772997, 0.20686604, 0.25426316, 0.31583686, 0.58379617,
                                         0.60854246, 0.70308652, 0.75609128, 0.74480276, 0.73980283, 0.71272712,
                                         0.62096204, 0.54032503, 0.50775975, 0.45727597, 0.42481463, 0.41557188,
                                         0.35181203, 0.34712741, 0.33773559, 0.3207941, 0.28726665, 0.26055698,
                                         0.24649891, 0.21356518])
        sp_en_trans_var_ratio = np.array([0.00212832, 0.00213949, 0.00279269, 0.00429538, 0.02493189, 0.03539891,
                                          0.08223719, 0.09505613, 0.10497895, 0.14144305, 0.19738455, 0.28717836,
                                          0.31987576, 0.37558719, 0.33298642, 0.29387486, 0.25336007, 0.24135965,
                                          0.22504063, 0.19960773, 0.17008746, 0.14751186, 0.14436817, 0.1428359,
                                          0.13761358, 0.13531406, 0.13319551, 0.12951574, 0.12117375, 0.1124996,
                                          0.11315788, 0.10922431])
        neg_sp_en_trans_var_ratio = np.array([0.00217094, 0.00219424, 0.00339651, 0.00554042, 0.02437975, 0.03625614,
                                              0.09760141, 0.12553468, 0.17186795, 0.22009408, 0.27778469, 0.41421039,
                                              0.47449628, 0.56308636, 0.51813834, 0.50776339, 0.51891065, 0.48373888,
                                              0.45025648, 0.4179804, 0.37028887, 0.33180174, 0.32681159, 0.314175,
                                              0.30494091, 0.30643241, 0.28818484, 0.27963693, 0.2585485, 0.2321744,
                                              0.22055458, 0.18973123])
        pivot_idx = 12
    else:
        print("=>=> {} is unrecognizable for getting "
              "the truthfulness ratio distribution...".format(args.model))
    assert (cities_var_ratio is not None
            and neg_cities_var_ratio is not None
            and sp_en_trans_var_ratio is not None
            and neg_sp_en_trans_var_ratio is not None
            and pivot_idx is not None)

    cities_var_ratio_pd = cities_var_ratio / np.sum(cities_var_ratio)
    neg_cities_var_ratio_pd = neg_cities_var_ratio / np.sum(neg_cities_var_ratio)
    sp_en_trans_var_ratio_pd = sp_en_trans_var_ratio / np.sum(sp_en_trans_var_ratio)
    neg_sp_en_trans_var_pd = neg_sp_en_trans_var_ratio / np.sum(neg_sp_en_trans_var_ratio)
    var_ratio_pd = (cities_var_ratio_pd + neg_cities_var_ratio_pd +
                    sp_en_trans_var_ratio_pd + neg_sp_en_trans_var_pd) / 4.0  # truthfulness ratio

    all_layer_ratio = np.full(layer_num, 1 - args.sparsity_ratio)
    # The scaling factor (e.g., 0.1) determines how much deviation from sparsity is allowed

    adjustments = (var_ratio_pd - 1 / layer_num) * scaling_factor * layer_num  # Center around 0
    all_layer_truthfulness_ratio = all_layer_ratio + adjustments

    return all_layer_truthfulness_ratio, pivot_idx


def prune_wanda(args, model, tokenizer, device, prepend_calibration,
                prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False

    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    args.nsamples = 64 + 64
    for calibration_data in args.calibration_datasets:
        """
        if calibration_data == 'c4':
            each_dataset_num = 64
        elif calibration_data == 'enriched_truth_qa':
            each_dataset_num = 64
        else:
            print("error!")
            exit(-1)
        """
        
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048]) 
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))


    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device, 
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)
    

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        """
        if f"model.layers.{i}" in model.hf_device_map:
            ## handle the case for llama-30B and llama-65B, 
            # when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (inps.to(dev), outs.to(dev),
                                                        attention_mask.to(dev), position_ids.to(dev))
        """
        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            """from .layerwrapper import WrappedGPT"""
            wrapped_layers[name] = WrappedGPT(subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """
            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp


        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        print("\n=>=>=>=> Start pruning...\n")
        for name in subset:
            print(f"\n=>=> Pruning layer {i} name {name}...")
            """
               The outlier score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = (torch.abs(subset[name].weight.data) *
                        torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            W_mask = (torch.zeros_like(W_metric) == 1)  # initialize a mask to be all False
            if prune_n != 0:
                print("\n=>=> Triggering {}:{} structured pruning...\n"
                      .format(prune_n, prune_m))
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp,
                                                           prune_n,
                                                           dim=1,
                                                           largest=False)[1],
                                        True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                """
                   torch.sort(input, dim, stable): This method sorts the elements 
                of the input tensor (W_metric) along the specified dimension 
                (dim=-1 means sorting along the last dimension, i.e., row-wise). 
                   The result of torch.sort() is a tuple where:
                • sort_res[0] contains the sorted values.
                • sort_res[1] contains the indices that would sort the original 
                array.
                • stable=True: The stable flag ensures that if two elements are 
                equal, their original order is preserved.
                   After this step, sort_res[1] will be a tensor containing the 
                indices that would sort the elements of W_metric row-wise.
                """

                if args.use_variant:  # Wanda variant in the appendix
                    print("\n=>=> Triggering wanda variant in the appendix...\n")
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = (
                        return_given_alpha(alpha, sort_res, W_metric,
                                           tmp_metric, sum_before)
                    )
                    while ((torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001)
                           and (alpha_hist[1] - alpha_hist[0] >= 0.001)):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric,
                                                                  tmp_metric, sum_before)
                    print(f"\n=>=>alpha found {alpha} "
                          f"sparsity {cur_sparsity:.6f}\n")
                else:  # Unstructured pruning
                    print("\n=>=> Triggering unstructured pruning...\n")
                    """
                       [:, :int(W_metric.shape[1] * layer_sparsity_ratio)]: 
                    This line selects the top fraction of indices from the 
                    sorted indices.
                    """
                    indices = sort_res[1][:, :int(W_metric.shape[1] *
                                                  args.sparsity_ratio)]
                    """
                       scatter_(dim, index, src): This is an in-place operation 
                    that “scatters” the values from the source tensor (src) into 
                    W_mask at the specified index positions along the dimension 
                    dim.
                    • dim=1: This specifies that the scattering happens along 
                    the second dimension (columns).
                    • indices: This is the tensor of indices where the values 
                    will be scattered. These indices correspond to the lowest 
                    values in W_metric, as obtained from the previous step.
                    • True: This is the source value (or tensor) that will be 
                    scattered into W_mask. In this case, the value True is 
                    scattered into the mask at the specified indices.
                    """
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # Set weights to zero

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]

        inps, outs = outs, inps  # for i in range(len(layers))

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print("\n=>=> Finish pruning!\n")


def prune_owl(args, model, tokenizer, device, prepend_calibration,
              prune_n=0, prune_m=0):
    ##### Step1: calculate outlier ratio
    print("\n=>=>=>=> Start calculating outlier ratio...")
    all_layer_ratio = []
    use_cache = model.config.use_cache 
    model.config.use_cache = False

    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    print("\n=>=> Start getting Layerwise Outlier Distribution (LOD)...")
    for i in range(len(layers)):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(module=layer, layers=[nn.Linear])
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """
        
        """
        if f"model.layers.{i}" in model.hf_device_map and ("30b" in args.model or "65b" in args.model):
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            """from .layerwrapper import WrappedGPT"""
            wrapped_layers[name] = WrappedGPT(layer=subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """
            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            print(f"\n=>=> Getting the score of layer {i} name {name}...")
            """
               The score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = (torch.abs(subset[name].weight.data) *
                        torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))  # Features
            print("=>=> W_metric.size() is {}".format(W_metric.size()))
            layer_wmetric.append(W_metric)    
                

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

        print("\n=>=> len(layer_wmetric) is {}".format(len(layer_wmetric)))
        """ Pruning one layer together based on the ratio.
        self_attn.q_proj: W_metric.size() is torch.Size([4096, 4096])
        self_attn.k_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.v_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.o_proj: W_metric.size() is torch.Size([4096, 4096])
        mlp.gate_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.up_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.down_proj: W_metric.size() is torch.Size([4096, 14336])
        """
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        print("=>=> layer_wmetric.size() is {}\n".format(layer_wmetric.size())) # layer_wmetric.size() is torch.Size([218103808])
        
        """
           Weight outliers are identified as weights whose outlier scores are 
        at least M times larger than the mean. Could the hyperparameter M be dynamically
        adjusted by the truth ratio?
        """
        for out_ratio in [args.Hyper_m]:
            print("\n=>=> Hyperparameter M is {}".format(out_ratio))
            """
            from prune_all import check_outlier_mean
               The “outlier ratio” of A by identifying 
            elements whose magnitude is M times greater 
            than the averaged value in each layer.
            """

            out_ratio_layer = check_outlier_mean(mask=layer_wmetric, threshold=out_ratio)
            print("=>=> outlier ratio is {}\n".format(out_ratio_layer))

        all_layer_ratio.append(out_ratio_layer)


    print("\n=>=> Finish getting Layerwise Outlier Distribution (LOD)!")


    print("\n=>=> Before adjustment, Layerwise Outlier "
          "Distribution (LOD) is {}".format(all_layer_ratio))
    """
       The hyperparameter λ is to constrain the layerwise 
    sparsity to fall within a small range, specifically, Si ∈ [S − λ, S + λ],
    """
    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min())
                       * (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    print("=> np.mean(all_layer_ratio) is {}".format(np.mean(all_layer_ratio)))
    print("=> np.max(all_layer_ratio) is {}".format(np.max(all_layer_ratio)))
    print("=> np.min(all_layer_ratio) is {}".format(np.min(all_layer_ratio)))
    print("=>=> After adjustment, all_layer_ratio is {}\n".format(all_layer_ratio))
    """ For --sparsity_ratio 0.7 --Lamda 0.08 --Hyper_m 5 all_layer_ratio is 
    [0.33274324 0.38586082 0.35804334 0.37758282 0.38562737 0.3924267
    0.3802845  0.36321172 0.36712167 0.36532444 0.35489393 0.32279734
    0.33466587 0.32759054 0.30860276 0.29964119 0.27494305 0.26779475
    0.26306604 0.24987249 0.24384089 0.2405093  0.23954733 0.23692139
    0.23730957 0.23506433 0.23603741 0.23549463 0.2324267  0.2334241
    0.23947312 0.27785662]
    """
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    print("\n=>=> Finish calculating outlier ratio!\n")

    ##### Step2: Pruning
    print("\n=>=>=>=> Start pruning...\n")
    use_cache = model.config.use_cache 
    model.config.use_cache = False
    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])


    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    print("\n=>=> Start layerwise pruning via LOD...")
    for i in range(len(layers)):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(layer)
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """
        """
        if f"model.layers.{i}" in model.hf_device_map:
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """
            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"\n=>=> Pruning layer {i} name {name}...")
            """
               The outlier score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = \
                (torch.abs(subset[name].weight.data) *
                 torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = (
                torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            print(f"=>=> The layer {i}'s sparsity_ratio is {layer_sparsity_ratio}.")
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize a mask to be all False
            print(f"=>=> W_mask.size() is {W_mask.size()}.")
            print(f"=>=> W_mask is {W_mask}.")
            if prune_n != 0:
                print("\n=>=> Triggering n:m structured pruning...\n")
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii+prune_m)].float()
                        W_mask.scatter_(1, ii+torch.topk(tmp,
                                                         prune_n,
                                                         dim=1,
                                                         largest=False)[1],
                                        True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                """
                   torch.sort(input, dim, stable): This method sorts the elements 
                of the input tensor (W_metric) along the specified dimension 
                (dim=-1 means sorting along the last dimension, i.e., row-wise). 
                   The result of torch.sort() is a tuple where:
                • sort_res[0] contains the sorted values.
                • sort_res[1] contains the indices that would sort the original 
                array.
                • stable=True: The stable flag ensures that if two elements are 
                equal, their original order is preserved.
                   After this step, sort_res[1] will be a tensor containing the 
                indices that would sort the elements of W_metric row-wise.
                """
                if args.use_variant:  # Wanda variant in the appendix
                    print("\n=>=> Triggering wanda variant in the appendix...\n")
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    """from prune_all import return_given_alpha"""
                    W_mask, cur_sparsity = (
                        return_given_alpha(alpha, sort_res, W_metric,
                                           tmp_metric, sum_before)
                    )
                    while ((torch.abs(cur_sparsity -
                                      layer_sparsity_ratio) > 0.001)
                           and (alpha_hist[1]-alpha_hist[0] >= 0.001)):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        """from prune_all import return_given_alpha"""
                        W_mask, cur_sparsity = (
                            return_given_alpha(alpha, sort_res, W_metric,
                                               tmp_metric, sum_before)
                        )
                    print(f"\n=>=>alpha found {alpha} "
                          f"sparsity {cur_sparsity:.6f}\n")
                else:  # Unstructured pruning
                    print("\n=>=> Triggering unstructured pruning...\n")
                    """
                       [:, :int(W_metric.shape[1] * layer_sparsity_ratio)]: 
                    This line selects the top fraction of indices from the 
                    sorted indices.
                    """
                    indices = sort_res[1][:, :int(W_metric.shape[1] *
                                                  layer_sparsity_ratio)]
                    """
                       scatter_(dim, index, src): This is an in-place operation 
                    that “scatters” the values from the source tensor (src) into 
                    W_mask at the specified index positions along the dimension 
                    dim.
                    • dim=1: This specifies that the scattering happens along 
                    the second dimension (columns).
                    • indices: This is the tensor of indices where the values 
                    will be scattered. These indices correspond to the lowest 
                    values in W_metric, as obtained from the previous step.
                    • True: This is the source value (or tensor) that will be 
                    scattered into W_mask. In this case, the value True is 
                    scattered into the mask at the specified indices.
                    """
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # set weights to zero

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print("\n=>=> Finish pruning!\n")


def prune_wanda_option1(args, model, tokenizer, device, prepend_calibration,
                        prune_n=0, prune_m=0):
    """
       The pruning ratio be arranged by truthfulness ratio.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    layer_num = len(layers)
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    """from prune_all import get_truthfulness_ratio"""
    all_layer_truthfulness_ratio, _ = get_truthfulness_ratio(args=args, layer_num=layer_num)
    sparsity_array = np.full(layer_num, 1) - all_layer_truthfulness_ratio
    print("\n=>=> sparsity_array: {}".format(sparsity_array))
    print("=>=> np.mean(sparsity_array): {}\n".format(np.mean(sparsity_array)))

    for i in range(layer_num):
        layer = layers[i]
        subset = find_layers(layer)
        """
        if f"model.layers.{i}" in model.hf_device_map:
            ## handle the case for llama-30B and llama-65B, 
            # when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (inps.to(dev), outs.to(dev),
                                                        attention_mask.to(dev), position_ids.to(dev))
        """
        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            """from .layerwrapper import WrappedGPT"""
            wrapped_layers[name] = WrappedGPT(subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """

            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        print("\n=>=>=>=> Start pruning...\n")
        for name in subset:
            print(f"\n=>=> Pruning layer {i} name {name}...")
            """
               The outlier score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = (torch.abs(subset[name].weight.data) *
                        torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            W_mask = (torch.zeros_like(W_metric) == 1)  # initialize a mask to be all False
            if prune_n != 0:
                print("\n=>=> Triggering {}:{} structured pruning...\n"
                      .format(prune_n, prune_m))
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp,
                                                           prune_n,
                                                           dim=1,
                                                           largest=False)[1],
                                        True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                """
                   torch.sort(input, dim, stable): This method sorts the elements 
                of the input tensor (W_metric) along the specified dimension 
                (dim=-1 means sorting along the last dimension, i.e., row-wise). 
                   The result of torch.sort() is a tuple where:
                • sort_res[0] contains the sorted values.
                • sort_res[1] contains the indices that would sort the original 
                array.
                • stable=True: The stable flag ensures that if two elements are 
                equal, their original order is preserved.
                   After this step, sort_res[1] will be a tensor containing the 
                indices that would sort the elements of W_metric row-wise.
                """

                if args.use_variant:  # Wanda variant in the appendix
                    print("\n=>=> Triggering wanda variant in the appendix...\n")
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = (
                        return_given_alpha(alpha, sort_res, W_metric,
                                           tmp_metric, sum_before)
                    )
                    while ((torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001)
                           and (alpha_hist[1] - alpha_hist[0] >= 0.001)):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric,
                                                                  tmp_metric, sum_before)
                    print(f"\n=>=>alpha found {alpha} "
                          f"sparsity {cur_sparsity:.6f}\n")
                else:  # Unstructured pruning
                    print("\n=>=> Triggering unstructured pruning...\n")
                    """
                       [:, :int(W_metric.shape[1] * layer_sparsity_ratio)]: 
                    This line selects the top fraction of indices from the 
                    sorted indices.
                    """
                    indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_array[i])]
                    """
                       scatter_(dim, index, src): This is an in-place operation 
                    that “scatters” the values from the source tensor (src) into 
                    W_mask at the specified index positions along the dimension 
                    dim.
                    • dim=1: This specifies that the scattering happens along 
                    the second dimension (columns).
                    • indices: This is the tensor of indices where the values 
                    will be scattered. These indices correspond to the lowest 
                    values in W_metric, as obtained from the previous step.
                    • True: This is the source value (or tensor) that will be 
                    scattered into W_mask. In this case, the value True is 
                    scattered into the mask at the specified indices.
                    """
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # Set weights to zero

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]

        inps, outs = outs, inps  # for i in range(len(layers))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n=>=> Finish pruning!\n")


def prune_owl_option2(args, model, tokenizer, device, prepend_calibration,
                      prune_n=0, prune_m=0):
    """
       Option 2: The pruning ratio be arranged by truthfulness ratio, then
    first ten layers (layer 0 to layer 9) are copied by OWL and smoothed
    to sparsity. Optionally, try averaging the middle ten layers
    (layer 10 to layer 19).
    """
    ##### Step1: calculate outlier ratio
    print("\n=>=>=>=> Start calculating outlier ratio...")
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    layer_num = len(layers)
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    print("\n=>=> Start getting Layerwise Outlier Distribution (LOD)...")
    for i in range(layer_num):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(module=layer, layers=[nn.Linear])
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """

        """
        if f"model.layers.{i}" in model.hf_device_map and ("30b" in args.model or "65b" in args.model):
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            """from .layerwrapper import WrappedGPT"""
            wrapped_layers[name] = WrappedGPT(layer=subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """

            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            print(f"\n=>=> Getting the score of layer {i} name {name}...")
            """
               The score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = (torch.abs(subset[name].weight.data) *
                        torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))  # Features
            print("=>=> W_metric.size() is {}".format(W_metric.size()))
            layer_wmetric.append(W_metric)

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

        print("\n=>=> len(layer_wmetric) is {}".format(len(layer_wmetric)))
        """ Pruning one layer together based on the ratio.
        self_attn.q_proj: W_metric.size() is torch.Size([4096, 4096])
        self_attn.k_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.v_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.o_proj: W_metric.size() is torch.Size([4096, 4096])
        mlp.gate_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.up_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.down_proj: W_metric.size() is torch.Size([4096, 14336])
        """
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        print("=>=> layer_wmetric.size() is {}\n".format(
            layer_wmetric.size()))  # layer_wmetric.size() is torch.Size([218103808])

        """
           Weight outliers are identified as weights whose outlier scores are 
        at least M times larger than the mean. Could the hyperparameter M be dynamically
        adjusted by the truth ratio?
        """
        for out_ratio in [args.Hyper_m]:
            print("\n=>=> Hyperparameter M is {}".format(out_ratio))
            """
            from prune_all import check_outlier_mean
               The “outlier ratio” of A by identifying 
            elements whose magnitude is M times greater 
            than the averaged value in each layer.
            """

            out_ratio_layer = check_outlier_mean(mask=layer_wmetric, threshold=out_ratio)
            print("=>=> outlier ratio is {}\n".format(out_ratio_layer))

        all_layer_ratio.append(out_ratio_layer)

    print("\n=>=> Finish getting Layerwise Outlier Distribution (LOD)!")

    print("\n=>=> Before adjustment, Layerwise Outlier "
          "Distribution (LOD) is {}".format(all_layer_ratio))
    """
       The hyperparameter λ is to constrain the layerwise 
    sparsity to fall within a small range, specifically, Si ∈ [S − λ, S + λ],
    """
    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min())
                       * (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2))
    
    owl_layer_maintain_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    print("=>=> After adjustment, all_layer_ratio is {}\n".format(owl_layer_maintain_ratio))
    """ For --sparsity_ratio 0.7 --Lamda 0.08 --Hyper_m 5 all_layer_ratio is 
    [0.33274324 0.38586082 0.35804334 0.37758282 0.38562737 0.3924267
    0.3802845  0.36321172 0.36712167 0.36532444 0.35489393 0.32279734
    0.33466587 0.32759054 0.30860276 0.29964119 0.27494305 0.26779475
    0.26306604 0.24987249 0.24384089 0.2405093  0.23954733 0.23692139
    0.23730957 0.23506433 0.23603741 0.23549463 0.2324267  0.2334241
    0.23947312 0.27785662]
    """
    """from prune_all import get_truthfulness_ratio"""
    all_layer_truthfulness_ratio, pivot_idx = (
        get_truthfulness_ratio(args=args, layer_num=layer_num)
    )
    option2_layer_maintain_ratio = copy.deepcopy(all_layer_truthfulness_ratio)
    """
       This copies the values from array2 into array1 for the 
    specified slice, but the arrays themselves do not share the 
    same reference.
    """
    option2_layer_maintain_ratio[0:pivot_idx] = owl_layer_maintain_ratio[0:pivot_idx]
    
    print("\n=>=> Before smoothing, option2_layer_maintain_ratio is {} "
          "and np.mean(option2_layer_maintain_ratio) is {}"
          .format(option2_layer_maintain_ratio,
                  np.mean(option2_layer_maintain_ratio)))
    density = 1 - args.sparsity_ratio

    option2_layer_maintain_ratio = (option2_layer_maintain_ratio -
                                    (np.mean(option2_layer_maintain_ratio) - density))
    """
    option2_layer_maintain_ratio[pivot_idx:] = (option2_layer_maintain_ratio[pivot_idx:] - 
                                         (np.mean(option2_layer_maintain_ratio) - density) 
                                         * option2_layer_maintain_ratio.shape[0] 
                                         / (option2_layer_maintain_ratio.shape[0] - 10))
    """
    print("=>=> After smoothing, option2_layer_maintain_ratio is {} "
          "and np.mean(option2_layer_maintain_ratio) is {}\n"
          .format(option2_layer_maintain_ratio,
                  np.mean(option2_layer_maintain_ratio)))


    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=>=> Finish calculating outlier ratio!\n")

    ##### Step2: Pruning
    print("\n=>=>=>=> Start pruning...\n")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    print("\n=>=> Start layerwise pruning via LOD...")
    for i in range(len(layers)):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(layer)
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """
        """
        if f"model.layers.{i}" in model.hf_device_map:
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """

            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"\n=>=> Pruning layer {i} name {name}...")
            """
               The outlier score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = \
                (torch.abs(subset[name].weight.data) *
                 torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = (
                torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            )

            layer_sparsity_ratio = 1 - option2_layer_maintain_ratio[i]
            print(f"=>=> The layer {i}'s sparsity_ratio is {layer_sparsity_ratio}.")
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize a mask to be all False
            print(f"=>=> W_mask.size() is {W_mask.size()}.")
            print(f"=>=> W_mask is {W_mask}.")
            if prune_n != 0:
                print("\n=>=> Triggering n:m structured pruning...\n")
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp,
                                                           prune_n,
                                                           dim=1,
                                                           largest=False)[1],
                                        True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                """
                   torch.sort(input, dim, stable): This method sorts the elements 
                of the input tensor (W_metric) along the specified dimension 
                (dim=-1 means sorting along the last dimension, i.e., row-wise). 
                   The result of torch.sort() is a tuple where:
                • sort_res[0] contains the sorted values.
                • sort_res[1] contains the indices that would sort the original 
                array.
                • stable=True: The stable flag ensures that if two elements are 
                equal, their original order is preserved.
                   After this step, sort_res[1] will be a tensor containing the 
                indices that would sort the elements of W_metric row-wise.
                """
                if args.use_variant:  # Wanda variant in the appendix
                    print("\n=>=> Triggering wanda variant in the appendix...\n")
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    """from prune_all import return_given_alpha"""
                    W_mask, cur_sparsity = (
                        return_given_alpha(alpha, sort_res, W_metric,
                                           tmp_metric, sum_before)
                    )
                    while ((torch.abs(cur_sparsity -
                                      layer_sparsity_ratio) > 0.001)
                           and (alpha_hist[1] - alpha_hist[0] >= 0.001)):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        """from prune_all import return_given_alpha"""
                        W_mask, cur_sparsity = (
                            return_given_alpha(alpha, sort_res, W_metric,
                                               tmp_metric, sum_before)
                        )
                    print(f"\n=>=>alpha found {alpha} "
                          f"sparsity {cur_sparsity:.6f}\n")
                else:  # Unstructured pruning
                    print("\n=>=> Triggering unstructured pruning...\n")
                    """
                       [:, :int(W_metric.shape[1] * layer_sparsity_ratio)]: 
                    This line selects the top fraction of indices from the 
                    sorted indices.
                    """
                    indices = sort_res[1][:, :int(W_metric.shape[1] *
                                                  layer_sparsity_ratio)]
                    """
                       scatter_(dim, index, src): This is an in-place operation 
                    that “scatters” the values from the source tensor (src) into 
                    W_mask at the specified index positions along the dimension 
                    dim.
                    • dim=1: This specifies that the scattering happens along 
                    the second dimension (columns).
                    • indices: This is the tensor of indices where the values 
                    will be scattered. These indices correspond to the lowest 
                    values in W_metric, as obtained from the previous step.
                    • True: This is the source value (or tensor) that will be 
                    scattered into W_mask. In this case, the value True is 
                    scattered into the mask at the specified indices.
                    """
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # set weights to zero

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n=>=> Finish pruning!\n")


def prune_owl_option3(args, model, tokenizer, device, prepend_calibration,
                                prune_n=0, prune_m=0):
    """
       Option 3: Middle layer, more truth capability, we need to
    make M lower to save more parameters.
    """
    ##### Step1: calculate outlier ratio
    print("\n=>=>=>=> Start calculating outlier ratio...")
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))



    # Option 2: Middle layer, more truth capability, we need to make M lower to save more parameters.
    # LLaMA-3.1-8B-instruct
    cities_var_ratio = np.array([3.39071090e-04, 1.04245816e-03, 2.39174780e-03, 6.37486248e-03,
                                 3.81021744e-02, 8.09538393e-02, 1.32550373e-01, 2.47270862e-01,
                                 3.81992400e-01, 5.26349089e-01, 6.45404196e-01, 7.53192202e-01,
                                 7.32511960e-01, 6.94884655e-01, 6.29495111e-01, 6.29789162e-01,
                                 5.79958105e-01, 5.05475324e-01, 4.78962535e-01, 4.64287028e-01,
                                 4.53694642e-01, 4.26050788e-01, 4.09579671e-01, 3.94378530e-01,
                                 3.39963516e-01, 3.15667607e-01, 2.97716687e-01, 2.58198872e-01,
                                 2.57569347e-01, 2.37331753e-01, 1.95439694e-01, 1.61557409e-01])
    neg_cities_var_ratio = np.array([2.75635858e-04, 1.04091549e-03, 2.34982485e-03, 6.99114158e-03,
                                     4.09611215e-02, 8.77295960e-02, 1.22624907e-01, 1.88092399e-01,
                                     3.41688715e-01, 5.12884356e-01, 6.27721497e-01, 6.55794854e-01,
                                     1.01014698e+00, 8.92597053e-01, 7.48149017e-01, 6.31548774e-01,
                                     5.36927071e-01, 4.01787252e-01, 3.04532688e-01, 2.92508502e-01,
                                     2.59050552e-01, 2.27964383e-01, 2.02185914e-01, 1.68580469e-01,
                                     9.77188368e-02, 9.09651150e-02, 7.93741087e-02, 6.23002706e-02,
                                     6.09152265e-02, 5.66922517e-02, 4.91064340e-02, 3.92692501e-02])
    sp_en_trans_var_ratio = np.array([0.00650565, 0.00233356, 0.00296728, 0.0100899, 0.01508244, 0.03669597,
                                      0.07735114, 0.12632234, 0.19591362, 0.24608271, 0.31299248, 0.3722659,
                                      0.38678397, 0.32486482, 0.27961429, 0.23536007, 0.21875916, 0.19583675,
                                      0.18217044, 0.17587753, 0.16802919, 0.14759805, 0.13368673, 0.12258105,
                                      0.11283833, 0.10460265, 0.0997852, 0.09395919, 0.08996084, 0.08321178,
                                      0.08815841, 0.11194301])
    neg_sp_en_trans_var_ratio = np.array([0.00566225, 0.00234005, 0.00331782, 0.0127961, 0.01657107, 0.03848792,
                                          0.09450669, 0.16633245, 0.26102221, 0.30004234, 0.3645685, 0.45129881,
                                          0.5066566, 0.47487872, 0.43538618, 0.38333991, 0.38493038, 0.38590715,
                                          0.35965402, 0.34839013, 0.34806121, 0.30753975, 0.26950751, 0.25464624,
                                          0.21906972, 0.19438002, 0.17653393, 0.15905798, 0.15261992, 0.14192056,
                                          0.12835094, 0.11994015])
    cities_var_ratio_pd = cities_var_ratio / np.sum(cities_var_ratio)
    neg_cities_var_ratio_pd = neg_cities_var_ratio / np.sum(neg_cities_var_ratio)
    sp_en_trans_var_ratio_pd = sp_en_trans_var_ratio / np.sum(sp_en_trans_var_ratio)
    neg_sp_en_trans_var_pd = neg_sp_en_trans_var_ratio / np.sum(neg_sp_en_trans_var_ratio)
    var_ratio_pd = (cities_var_ratio_pd + neg_cities_var_ratio_pd +
                    sp_en_trans_var_ratio_pd + neg_sp_en_trans_var_pd) / 4.0  # truthfulness ratio

    hyper_m = args.Hyper_m
    layer_num = len(layers)
    all_layer_hyper_m = np.full(layer_num, hyper_m)

    # The scaling factor (e.g., 0.1) determines how much deviation from sparsity is allowed
    scaling_factor = 1.0
    adjustments = (var_ratio_pd - 1 / 32) * scaling_factor * 32  # Center around 0
    all_layer_hyper_m = all_layer_hyper_m - adjustments
    print("\n=>=> all_layer_hyper_m: {}".format(all_layer_hyper_m))
    print("=>=> np.mean(all_layer_hyper_m): {}\n".format(np.mean(all_layer_hyper_m)))

    print("\n=>=> Start getting Layerwise Outlier Distribution (LOD)...")
    for i in range(layer_num):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(module=layer, layers=[nn.Linear])
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """

        """
        if f"model.layers.{i}" in model.hf_device_map and ("30b" in args.model or "65b" in args.model):
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            """from .layerwrapper import WrappedGPT"""
            wrapped_layers[name] = WrappedGPT(layer=subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """

            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            print(f"\n=>=> Getting the score of layer {i} name {name}...")
            """
               The score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = (torch.abs(subset[name].weight.data) *
                        torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))  # Features
            print("=>=> W_metric.size() is {}".format(W_metric.size()))
            layer_wmetric.append(W_metric)

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

        print("\n=>=> len(layer_wmetric) is {}".format(len(layer_wmetric)))
        """ Pruning one layer together based on the ratio.
        self_attn.q_proj: W_metric.size() is torch.Size([4096, 4096])
        self_attn.k_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.v_proj: W_metric.size() is torch.Size([1024, 4096])
        self_attn.o_proj: W_metric.size() is torch.Size([4096, 4096])
        mlp.gate_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.up_proj: W_metric.size() is torch.Size([14336, 4096])
        mlp.down_proj: W_metric.size() is torch.Size([4096, 14336])
        """
        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        print("=>=> layer_wmetric.size() is {}\n".format(
            layer_wmetric.size()))  # layer_wmetric.size() is torch.Size([218103808])

        """
           Weight outliers are identified as weights whose outlier scores are 
        at least M times larger than the mean. Could the hyperparameter M be dynamically
        adjusted by the truth ratio?
        """
        for out_ratio in [all_layer_hyper_m[i]]:
            print("\n=>=> Hyperparameter M is {}".format(out_ratio))
            """
            from prune_all import check_outlier_mean
               The “outlier ratio” of A by identifying 
            elements whose magnitude is M times greater 
            than the averaged value in each layer.
            """

            out_ratio_layer = check_outlier_mean(mask=layer_wmetric, threshold=out_ratio)
            print("=>=> outlier ratio is {}\n".format(out_ratio_layer))

        all_layer_ratio.append(out_ratio_layer)

    print("\n=>=> Finish getting Layerwise Outlier Distribution (LOD)!")

    print("\n=>=> Before adjustment, Layerwise Outlier "
          "Distribution (LOD) is {}".format(all_layer_ratio))
    """
       The hyperparameter λ is to constrain the layerwise 
    sparsity to fall within a small range, specifically, Si ∈ [S − λ, S + λ],
    """
    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min())
                       * (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)
    print("=> np.mean(all_layer_ratio) is {}".format(np.mean(all_layer_ratio)))
    print("=> np.max(all_layer_ratio) is {}".format(np.max(all_layer_ratio)))
    print("=> np.min(all_layer_ratio) is {}".format(np.min(all_layer_ratio)))
    print("=>=> After adjustment, all_layer_ratio is {}\n".format(all_layer_ratio))
    """ For --sparsity_ratio 0.7 --Lamda 0.08 --Hyper_m 5 all_layer_ratio is 
    [0.33274324 0.38586082 0.35804334 0.37758282 0.38562737 0.3924267
    0.3802845  0.36321172 0.36712167 0.36532444 0.35489393 0.32279734
    0.33466587 0.32759054 0.30860276 0.29964119 0.27494305 0.26779475
    0.26306604 0.24987249 0.24384089 0.2405093  0.23954733 0.23692139
    0.23730957 0.23506433 0.23603741 0.23549463 0.2324267  0.2334241
    0.23947312 0.27785662]
    """
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("\n=>=> Finish calculating outlier ratio!\n")

    ##### Step2: Pruning
    print("\n=>=>=>=> Start pruning...\n")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("\n=>=> Loading {} calibration data...".format(args.calibration_datasets))
    dataloader = []
    assert args.nsamples % len(args.calibration_datasets) == 0
    each_dataset_num = args.nsamples // len(args.calibration_datasets)
    for calibration_data in args.calibration_datasets:
        assert model.seqlen == args.seqlen
        seqlen = args.seqlen
        if prepend_calibration is not None:
            print("=>=> prepend_calibration is not None")
            seqlen -= prepend_calibration.size()[1]
        """from .data import get_loaders"""
        print("\n=>=> Getting the {} calibration data loader...\n".format(calibration_data))
        curr_dataloader, _, pad_token = (
            get_loaders(args=args, data_name=calibration_data,
                        nsamples=each_dataset_num,
                        seqlen=seqlen,
                        tokenizer=tokenizer,
                        data_seqlen=None))
        print("\n=>=> Finish getting the {} calibration data loader!\n".format(calibration_data))
        dataloader.extend(curr_dataloader)

    assert len(dataloader) == args.nsamples
    for element in dataloader:
        assert element[0].size() == torch.Size([1, 2048]) and element[1].size() == torch.Size([1, 2048])
    print("=>=> Finish loading {} calibration data!\n".format(args.calibration_datasets))

    with (torch.no_grad()):
        inps, outs, attention_mask, position_ids = \
            prepare_calibration_input(args=args, model=model,
                                      dataloader=dataloader, device=device,
                                      pad_token=pad_token,
                                      prepend_calibration=prepend_calibration)

    print("\n=>=> inps.size() is {}"
          .format(inps.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> inps is {}".format(inps))  # meaningful
    print("\n=>=> outs.size() is {}"
          .format(outs.size()))  # torch.Size([128, 2048, 4096])
    print("=>=> outs is {}".format(outs))  # all zeros
    if attention_mask is not None:
        print("\n=>=> attention_mask.size() is {}"
              .format(attention_mask.size()))  # torch.Size([1, 1, 2048, 2048])

    print("=>=> attention_mask is {}".format(attention_mask))
    """
    tensor([[[[     0., -65504., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0., -65504.,  ..., -65504., -65504., -65504.],
      [     0.,      0.,      0.,  ..., -65504., -65504., -65504.],
      ...,
      [     0.,      0.,      0.,  ...,      0., -65504., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0., -65504.],
      [     0.,      0.,      0.,  ...,      0.,      0.,      0.]]]],
   device='cuda:0', dtype=torch.float16)
    """
    if position_ids is not None:
        print("\n=>=> position_ids.size() is {}"
              .format(position_ids.size()))  # torch.Size([1, 2048])

    if "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("\n=>=> type(layers) is {}".format(type(layers)))
    #  <class 'torch.nn.modules.container.ModuleList'>
    print("=>=> layers is {}\n".format(layers))

    print("\n=>=> Start layerwise pruning via LOD...")
    for i in range(len(layers)):
        print("\n=>=> layers[{}] name is {}\n".format(i, layers[i]))
        layer = layers[i]
        """ layer is 
           LlamaDecoderLayer(
        (self_attn): LlamaAttention(
           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
           (rotary_emb): LlamaRotaryEmbedding()
           )
        (mlp): LlamaMLP(
           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
           (act_fn): SiLUActivation()
           )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
        )
        """

        # subset is a dict to find specific types of layers
        subset = find_layers(layer)
        print(f"\n=>=> subset is {subset}\n")
        """subset is {
        'self_attn.q_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.k_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.v_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'self_attn.o_proj': Linear(in_features=4096, out_features=4096, bias=False), 
        'mlp.gate_proj': Linear(in_features=4096, out_features=11008, bias=False), 
        'mlp.down_proj': Linear(in_features=11008, out_features=4096, bias=False), 
        'mlp.up_proj': Linear(in_features=4096, out_features=11008, bias=False)
        }
        """
        """
        if f"model.layers.{i}" in model.hf_device_map:
            ## handle the case for llama-30B and llama-65B,
            # when the device map has multiple GPUs;
            print("\n=>=> Your model size is >= 30B...\n")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = \
                (inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev))
        """

        wrapped_layers = {}
        print(f"\n=>=> You are initializing the the wrapped layers...")
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        print(f"\n=>=> wrapped_layers is {wrapped_layers}\n")
        print(f"=>=> Finish initializing the wrapped layers!\n")

        def add_batch(name):
            """
               This function takes a name, which corresponds to
            the name of the layer. For each layer, it returns a
            function (tmp) that will be used as the hook.
            """

            def tmp(_, inp, out):
                """
                   The actual hook function that will be
                called during the forward pass of the model.
                This calls the add_batch method of the WrappedGPT
                instance corresponding to the layer name, passing in
                the input and output data. This likely updates some
                internal statistics related to the layer’s input
                and output.
                """
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []  # Initializes an empty list to store the hook handles
        for name in wrapped_layers:
            """
               subset[name].register_forward_hook(add_batch(name)):
            • This attaches a forward hook to the layer subset[name] using PyTorch’s 
            register_forward_hook method.
            • The hook function is generated by the add_batch(name) call, which creates 
            a customized hook for the layer with the specified name.
            • The hook will trigger every time the model performs a forward pass, 
            allowing the add_batch function to capture and process the inputs and 
            outputs of the layer.
               In PyTorch, a hook is a function that you can attach to a layer 
            (or a module). This hook gets called at certain points during the 
            model’s forward or backward pass. 
               There are different types of hooks, but the most common are:
            • Forward hooks: Triggered during the forward pass, after the layer 
            has processed its input and produced an output.
            •Backward hooks: Triggered during the backward pass when gradients are 
            being calculated.
               In your code, register_forward_hook is used, which attaches a function 
            to the layer that will be called whenever a forward pass is done. 
            Specifically, register_forward_hook attaches a function to a layer, 
            which is executed right after the forward pass of that layer, 
            allowing you to monitor or manipulate the inputs and outputs. To tie 
            it back to daily life: the hook is like a sensor or observer that gets 
            triggered when something happens (like a layer completing its forward
            pass) and lets you record or react to that event without interrupting 
            the main task.
            """
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"\n=>=> Pruning layer {i} name {name}...")
            """
               The outlier score of weight (W_metric) is calculated as the 
            accumulation of all input features connected to that weight, 
            multiplied by its magnitude, which also serves as the pruning 
            metric used by Wanda.
            """
            W_metric = \
                (torch.abs(subset[name].weight.data) *
                 torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            activation_data = (
                torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            )

            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            print(f"=>=> The layer {i}'s sparsity_ratio is {layer_sparsity_ratio}.")
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  # Initialize a mask to be all False
            print(f"=>=> W_mask.size() is {W_mask.size()}.")
            print(f"=>=> W_mask is {W_mask}.")
            if prune_n != 0:
                print("\n=>=> Triggering n:m structured pruning...\n")
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp,
                                                           prune_n,
                                                           dim=1,
                                                           largest=False)[1],
                                        True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                """
                   torch.sort(input, dim, stable): This method sorts the elements 
                of the input tensor (W_metric) along the specified dimension 
                (dim=-1 means sorting along the last dimension, i.e., row-wise). 
                   The result of torch.sort() is a tuple where:
                • sort_res[0] contains the sorted values.
                • sort_res[1] contains the indices that would sort the original 
                array.
                • stable=True: The stable flag ensures that if two elements are 
                equal, their original order is preserved.
                   After this step, sort_res[1] will be a tensor containing the 
                indices that would sort the elements of W_metric row-wise.
                """
                if args.use_variant:  # Wanda variant in the appendix
                    print("\n=>=> Triggering wanda variant in the appendix...\n")
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    """from prune_all import return_given_alpha"""
                    W_mask, cur_sparsity = (
                        return_given_alpha(alpha, sort_res, W_metric,
                                           tmp_metric, sum_before)
                    )
                    while ((torch.abs(cur_sparsity -
                                      layer_sparsity_ratio) > 0.001)
                           and (alpha_hist[1] - alpha_hist[0] >= 0.001)):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        """from prune_all import return_given_alpha"""
                        W_mask, cur_sparsity = (
                            return_given_alpha(alpha, sort_res, W_metric,
                                               tmp_metric, sum_before)
                        )
                    print(f"\n=>=>alpha found {alpha} "
                          f"sparsity {cur_sparsity:.6f}\n")
                else:  # Unstructured pruning
                    print("\n=>=> Triggering unstructured pruning...\n")
                    """
                       [:, :int(W_metric.shape[1] * layer_sparsity_ratio)]: 
                    This line selects the top fraction of indices from the 
                    sorted indices.
                    """
                    indices = sort_res[1][:, :int(W_metric.shape[1] *
                                                  layer_sparsity_ratio)]
                    """
                       scatter_(dim, index, src): This is an in-place operation 
                    that “scatters” the values from the source tensor (src) into 
                    W_mask at the specified index positions along the dimension 
                    dim.
                    • dim=1: This specifies that the scattering happens along 
                    the second dimension (columns).
                    • indices: This is the tensor of indices where the values 
                    will be scattered. These indices correspond to the lowest 
                    values in W_metric, as obtained from the previous step.
                    • True: This is the source value (or tensor) that will be 
                    scattered into W_mask. In this case, the value True is 
                    scattered into the mask at the specified indices.
                    """
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # set weights to zero

        for idx in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx])[0]
                else:
                    outs[idx] = layer(inps[idx].unsqueeze(0),
                                      attention_mask=attention_mask[idx],
                                      position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("\n=>=> Finish pruning!\n")
