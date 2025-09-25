import umap
import torch
import random
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from probes import TTPD, LRProbe
from utils import (DataManager, dataset_sizes, collect_training_data,
                   compute_statistics, compute_average_accuracies)


seed = 1000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Turn off to ensure deterministic behavior


def visualize_layerwise_probe_accuracy():
    # LLaMA3.1-8B-Chat

    layer_num = 32
    # Dummy data for LLaMA-7B
    layers = np.arange(1, layer_num + 1)
    filename = "probe_accuracies_layerwise.pkl"
    with open(filename, "rb") as f:
        probe_accuracies_layerwise = pickle.load(f)

    print(len(probe_accuracies_layerwise))
    print(probe_accuracies_layerwise)
    TTPD_layerwise_probe_accuracy = []
    LR_layerwise_probe_accuracy = []
    for layer_i_probe in probe_accuracies_layerwise:
        TTPD_layerwise_probe_accuracy.append(layer_i_probe['TTPD']['mean'])
        LR_layerwise_probe_accuracy.append(layer_i_probe['LRProbe']['mean'])


    # Colors matching the original plot
    TTPD_color = '#c6dcab'  # light green for option1
    LR_color = '#add8e6'  # light blue for option2

    # Common y-axis limits and ticks
    y_min, y_max = 0.4, 1.0
    y_ticks = np.arange(y_min, y_max + 0.2, 0.2)

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))


    # Plotting
    ax.plot(layers, TTPD_layerwise_probe_accuracy, color=TTPD_color,
            label='TTPD', linewidth=2, marker='o')
    ax.plot(layers, LR_layerwise_probe_accuracy, color=LR_color,
            label='LR', linewidth=2, marker='s')

    # Title and labels
    ax.set_title('LLaMA3.1-8B-Instruct', fontsize=14)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Probing Accuracy', fontsize=12)

    # Set y-axis limits and ticks
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()



def run_step1(train_sets, train_set_sizes, model_family,
              model_size, model_type, layer, device):
    # compare TTPD, LR and CCS on topic-specific datasets
    """from probes import TTPD, LRProbe"""
    probe_types = [TTPD, LRProbe]
    results = {TTPD: defaultdict(list),
               LRProbe: defaultdict(list)}
    num_iter = 20

    total_iterations = len(probe_types) * num_iter * len(train_sets)
    with tqdm(total=total_iterations,
              desc="Training and evaluating "
                   "classifiers") as pbar:  # progress bar
        """from probes import CCSProbe, TTPD, LRProbe, MMProbe"""
        for probe_type in probe_types:
            for n in range(num_iter):
                indices = np.arange(0, 12, 2)
                for i in indices:
                    """
                       Get a new NumPy array with the specified
                    elements removed for cross-validation training
                    data.
                    """
                    cv_train_sets = np.delete(np.array(train_sets),
                                              [i, i + 1], axis=0)

                    ## load training data
                    """
                       from utils import collect_training_data
                    polarity = -1.0 if 'neg_' in dataset_name else 1.0 
                    - acts_centered: torch.Size([1640, 4096]), abstract by 
                    mean
                    - acts: torch.Size([1640, 4096])
                    - labels: torch.Size([1640])
                    - polarities: torch.Size([1640]) 
                    """
                    acts_centered, acts, labels, polarities = \
                        collect_training_data(dataset_names=cv_train_sets,
                                              train_set_sizes=train_set_sizes,
                                              model_family=model_family,
                                              model_size=model_size,
                                              model_type=model_type,
                                              layer=layer)
                    print("=> acts_centered.size(): {}\nacts.size(): {}"
                          "\nlabels.size()：{}\npolarities.size()：{}"
                          .format(acts_centered.size(), acts.size(),
                                  labels.size(), polarities.size()))
                    if probe_type == TTPD:
                        """from probes import TTPD"""
                        probe = TTPD.from_data(acts_centered=acts_centered,
                                               acts=acts, labels=labels,
                                               polarities=polarities)
                    if probe_type == LRProbe:
                        """from probes import LRProbe
                           Logistic Regression (LR): Used by Burns et al. 
                        [2023] and Marks and Tegmark [2023] to classify 
                        statements as true or false based on internal 
                        model activations and by Li et al. [2024] to find 
                        truthful directions.
                        """
                        probe = LRProbe.from_data(acts, labels)


                    # Evaluate classification accuracy on held out datasets
                    dm = DataManager()
                    for j in range(0, 2):
                        dm.add_dataset(train_sets[i + j], model_family,
                                       model_size, model_type, layer,
                                       split=None, center=False, device='cpu')
                        acts, labels = dm.data[train_sets[i + j]]

                        predictions = probe.pred(acts)
                        results[probe_type][train_sets[i + j]].append(
                            (predictions == labels).float().mean().item()
                        )
                        pbar.update(1)

   
    """from utils import compute_statistics"""
    stat_results = compute_statistics(results=results)
    print("\n =>=> stat_results is:\n{}\n".format(stat_results))

    # Compute mean accuracies and standard deviations for each probe type
    """from utils import compute_average_accuracies"""
    probe_accuracies = compute_average_accuracies(results=results,
                                                  num_iter=num_iter)
    for probe_type, stats in probe_accuracies.items():
        print(f"\n=>=> {probe_type}:")
        print(f"=> Mean Accuracy: {stats['mean'] * 100:.2f}%")
        print(f"=> Standard Deviation of the mean accuracy: "
              f"{stats['std_dev'] * 100:.2f}%\n")

    return probe_accuracies


def visualize_latent_space(train_sets, train_set_sizes, model_family,
                           model_size, model_type, layer):
    ## load training data
    """
       from utils import collect_training_data
    polarity = -1.0 if 'neg_' in dataset_name else 1.0
    - acts_centered: torch.Size([1640, 4096]), abstract by
    mean
    - acts: torch.Size([1640, 4096])
    - labels: torch.Size([1640])
    - polarities: torch.Size([1640])
    """
    acts_centered, acts, labels, polarities = \
        collect_training_data(dataset_names=np.array(train_sets),
                              train_set_sizes=train_set_sizes,
                              model_family=model_family,
                              model_size=model_size,
                              model_type=model_type,
                              layer=layer)
    print("=> acts_centered.size(): {}\nacts.size(): {}"
          "\nlabels.size()：{}\npolarities.size()：{}"
          .format(acts_centered.size(), acts.size(),
                  labels.size(), polarities.size()))
    print(torch.min(acts), torch.max(acts))
    print(labels)

    # Assume your data is provided as PyTorch tensors
    # acts: torch.Size([1968, 4096]), labels: torch.Size([1968])
    # Example placeholder (replace with your actual data):
    # acts = torch.randn(1968, 4096)
    # labels = torch.randint(0, 2, (1968,))

    acts_np = acts.cpu().numpy()
    labels_np = labels.cpu().numpy()

    """ Normalize data (Standardization)
       Scales each feature to have:
    Mean = 0, Standard deviation = 1.
    """
    scaler = StandardScaler()
    acts_normalized = scaler.fit_transform(acts_np)

    # --- t-SNE Visualization ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    acts_tsne = tsne.fit_transform(acts_normalized)

    plt.figure(figsize=(8, 6))
    plt.scatter(acts_tsne[:, 0], acts_tsne[:, 1], c=labels_np, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Label')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

    # --- UMAP Visualization ---
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    acts_umap = reducer.fit_transform(acts_normalized)

    plt.figure(figsize=(8, 6))
    plt.scatter(acts_umap[:, 0], acts_umap[:, 1], c=labels_np, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Label')
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True)
    plt.show()

    # --- PCA Visualization ---
    pca = PCA(n_components=2)
    acts_pca = pca.fit_transform(acts_normalized)

    plt.figure(figsize=(8, 6))
    plt.scatter(acts_pca[:, 0], acts_pca[:, 1], c=labels_np, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Label')
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

    # --- Isomap Visualization ---
    isomap = Isomap(n_neighbors=10, n_components=2)
    acts_isomap = isomap.fit_transform(acts_normalized)

    plt.figure(figsize=(8, 6))
    plt.scatter(acts_isomap[:, 0], acts_isomap[:, 1], c=labels_np, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Label')
    plt.title('Isomap Visualization')
    plt.xlabel('Isomap Component 1')
    plt.ylabel('Isomap Component 2')
    plt.grid(True)
    plt.show()





def run_step2(train_sets, train_set_sizes, model_family,
              model_size, model_type, layer, device):
    """
       Generalisation to logical conjunctions and disjunctions. Compare
    TTPD, LR, CCS and MM on logical conjunctions and disjunctions.
    """
    val_sets = ["cities_conj", "cities_disj", "sp_en_trans_conj",
                "sp_en_trans_disj", "inventors_conj", "inventors_disj",
                "animal_class_conj", "animal_class_disj",
                "element_symb_conj", "element_symb_disj", "facts_conj",
                "facts_disj", "common_claim_true_false",
                "counterfact_true_false"]

    probe_types = [TTPD, LRProbe]
    results = {TTPD: defaultdict(list),
               LRProbe: defaultdict(list)}
    num_iter = 20

    total_iterations = len(probe_types) * num_iter
    with tqdm(total=total_iterations,
              desc="Training and evaluating "
                   "classifiers") as pbar:  # progress bar
        """from probes import CCSProbe, TTPD, LRProbe, MMProbe"""
        for probe_type in probe_types:
            for n in range(num_iter):
                # load training data
                """polarity = -1.0 if 'neg_' in dataset_name else 1.0"""
                acts_centered, acts, labels, polarities = \
                    collect_training_data(dataset_names=train_sets,
                                          train_set_sizes=train_set_sizes,
                                          model_family=model_family,
                                          model_size=model_size,
                                          model_type=model_type,
                                          layer=layer)
                if probe_type == TTPD:
                    """from probes import TTPD"""
                    probe = TTPD.from_data(acts_centered=acts_centered,
                                           acts=acts, labels=labels,
                                           polarities=polarities)
                if probe_type == LRProbe:
                    """from probes import LRProbe
                       Logistic Regression (LR): Used by Burns et al. [2023] 
                    and Marks and Tegmark [2023] to classify statements 
                    as true or false based on internal model activations 
                    and by Li et al. [2024] to find truthful directions.
                    """
                    probe = LRProbe.from_data(acts, labels)

                # evaluate classification accuracy on validation datasets
                dm = DataManager()
                for val_set in val_sets:
                    dm.add_dataset(val_set, model_family, model_size,
                                   model_type, layer, split=None,
                                   center=False,
                                   device='cpu')
                    acts, labels = dm.data[val_set]

                    # classifier specific predictions
                    predictions = probe.pred(acts)
                    results[probe_type][val_set].append(
                        (predictions == labels).float().mean().item()
                    )
                pbar.update(1)

    """from utils import compute_statistics"""
    stat_results = compute_statistics(results)

    # Compute mean accuracies and standard deviations for each probe type
    """from utils import compute_average_accuracies)"""
    probe_accuracies = compute_average_accuracies(results, num_iter)

    for probe_type, stats in probe_accuracies.items():
        print(f"\n=>=> {probe_type}:")
        print(f"=> Mean Accuracy: {stats['mean'] * 100:.2f}%")
        print(f"=> Standard Deviation of the mean accuracy: "
              f"{stats['std_dev'] * 100:.2f}%\n")


def main():
    # hyperparameters
    model_family = 'Llama3'  # options are 'Llama3', 'Llama2', 'Gemma', 'Gemma2' or 'Mistral'
    model_size = '8B'
    model_type = 'chat'  # options are 'chat' or 'base'
    model_type_list = ['c4_wanda0.5', 'c4_owl0.5', 'c4_wanda_option1_0.5', 'c4_owl_option2_0.5_0.04',
                       'c4_truthqa_wanda0.5', 'c4_truthqa_owl0.5', 'c4_truthqa_wanda_option1_0.5',
                       'c4_truthqa_owl_option2_0.5_0.04']

    model_type_list = ['c4_owl_option2_0.5_0.04']

    layer_num = 32
    layer = 12  # layer from which to extract activations
    run_step = {"step1": False, "step2": False}

    # gpu speeds up CCS training a fair bit but is not required
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # define datasets used for training
    train_sets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", 
                  "inventors", "neg_inventors", "animal_class", "neg_animal_class", 
                  "element_symb", "neg_element_symb", "facts", "neg_facts"]
    # get size of each training dataset to include an equal
    # number of statements from each topic in training data
    train_set_sizes = dataset_sizes(train_sets)

    if run_step["step1"]:
        """
           Figure 6 (a): Generalization accuracies of TTPD and LR 
        on topic-specific datasets:
        ["cities", "neg_cities", "sp_en_trans",
         "neg_sp_en_trans", "inventors", "neg_inventors",
         "animal_class", "neg_animal_class", "element_symb",
         "neg_element_symb", "facts", "neg_facts"]
           Mean and standard deviation computed from 20 training runs, 
        each on a different random sample of the training data.
        """
        print("\n=>=> You are running the step 1...\n")
        for model_type in model_type_list:
            print("\n=> For {}...".format(model_type))
            probe_accuracies_layerwise = []
            for layer in range(0, layer_num):
                print("\n=> For {}...".format(layer))
                probe_accuracies = run_step1(train_sets=train_sets,
                                             train_set_sizes=train_set_sizes,
                                             model_family=model_family, model_size=model_size,
                                             model_type=model_type, layer=layer, device=device)
                print("=> probe_accuracies: {}".format(probe_accuracies))
                probe_accuracies_layerwise.append(probe_accuracies)

            filename = model_type + "_probe_accuracies_layerwise.pkl"
            with open(filename, "wb") as f:
                pickle.dump(probe_accuracies_layerwise, f)
        print("\n=>=> Finish running the step 1!\n")

    visualize_layerwise_probe_accuracy()

    for layer in range(0, layer_num):
        print("\n=> Visualizing layer {}...".format(layer))
        layer = 12
        visualize_latent_space(train_sets=train_sets,
                               train_set_sizes=train_set_sizes,
                               model_family=model_family, model_size=model_size,
                               model_type=model_type, layer=layer)
        exit(-1)



    if run_step["step2"]:
        """
           Generalisation to logical conjunctions and disjunctions:
         ["cities_conj", "cities_disj", "sp_en_trans_conj",
          "sp_en_trans_disj", "inventors_conj", "inventors_disj",
          "animal_class_conj", "animal_class_disj",
          "element_symb_conj", "element_symb_disj", "facts_conj",
          "facts_disj", "common_claim_true_false",
          "counterfact_true_false"]
           Compare TTPD, LR, CCS and MM on logical conjunctions and 
        disjunctions.
        """
        print("\n=>=> You are running the step 2...\n")
        run_step2(train_sets=train_sets, train_set_sizes=train_set_sizes,
                  model_family=model_family, model_size=model_size,
                  model_type=model_type, layer=layer, device=device)
        print("\n=>=> Finish running the step 2!\n")


    return





if __name__ == '__main__':
    main()

