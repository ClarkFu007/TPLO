"""
   Code for generating figures from the first four paper sections;
from learning truth directions to exploring the dimensionality of the
truth subspace. You need to generate the following activations to run
this notebook (e.g. for Llama3-8B-Instruct):
python3 generate_acts.py \
--model_family Llama3 \
--model_size 8B \
--model_type chat \
--layers 12 \
--datasets all_topic_specific \
--device cuda:0

and

python3 generate_acts.py \
--model_family Llama3 \
--model_size 8B \
--model_type chat \
--layers -1 \
--datasets cities neg_cities sp_en_trans neg_sp_en_trans \
--device cuda:0
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LinearSegmentedColormap
# Local Python files
from probes import learn_truth_directions
from utils import (DataManager, dataset_sizes, collect_training_data,
                   compute_statistics)


def run_step1(model_family, model_size, model_type):
    """
       Find the layer with the largest separation between true and
    false statements. You need to have stored the activations in layers
    1-27 for all four datasets to run this cell. (Figure 2 in the paper).
    """
    layers = np.arange(1, 27, 1)
    datasets_separation = ['cities', 'neg_cities',
                           'sp_en_trans', 'neg_sp_en_trans']
    for dataset in datasets_separation:
        between_class_variances = []
        within_class_variances = []
        for layer_nr in layers:
            """from utils import DataManager"""
            dm = DataManager()
            dm.add_dataset(dataset_name=dataset, model_family=model_family,
                           model_size=model_size, model_type=model_type,
                           layer=layer_nr, split=None, center=False,
                           scale=False, device='cpu')
            acts, labels = dm.data[dataset]

            ## Calculate means for each class
            false_stmnt_ids = labels == 0
            true_stmnt_ids = labels == 1

            false_acts = acts[false_stmnt_ids]
            true_acts = acts[true_stmnt_ids]

            mean_false = false_acts.mean(dim=0)
            mean_true = true_acts.mean(dim=0)

            ## Calculate within-class variance
            within_class_variance_false = false_acts.var(dim=0).mean()
            within_class_variance_true = true_acts.var(dim=0).mean()
            within_class_variances.append(
                (within_class_variance_false
                 + within_class_variance_true).item() / 2
            )

            ## Calculate between-class variance
            overall_mean = acts.mean(dim=0)
            between_class_variances.append(((mean_false - overall_mean).pow(2)
                                            + (mean_true - overall_mean).pow(2))
                                           .mean().item() / 2)

        ratio_array = np.array(between_class_variances) / np.array(within_class_variances)

        array_name = dataset + "_" + model_type + "_ratio.npy"
        np.save(array_name, ratio_array)
        # ratio_array.shape: (26,)
        print("=> Dataset: {}, Model Family: {}, "
              "Model Size: {}, Model Type: {}, Silent Layer: {} / {}"
              .format(dataset, model_family, model_size, model_type,
                      np.argmax(ratio_array) + 1, ratio_array.shape[0]))

        plt.plot(layers, ratio_array, label=dataset)
    plt.legend(fontsize=16)
    plt.ylabel('Between class variance /'
               '\nwithin-class variance', fontsize=14)
    plt.xlabel('Layer', fontsize=14)
    plt.title('Separation between true and false'
              '\nstatements across layers', fontsize=15)
    plt.grid(True)
    plt.show()


def run_step2(train_sets, train_set_sizes, model_family,
              model_size, model_type, layer):
    """
       Supervised learning of the truth directions and
    classification accuracies.
    """
    nr_runs = 10
    results = {'t_g': defaultdict(list),
               't_p': defaultdict(list),
               'd_{LR}': defaultdict(list)}

    for _ in range(nr_runs):
        for i in range(0, len(train_sets), 2):
            # leave one dataset out (affirmative + negated)
            cv_train_sets = [j_set for j, j_set in enumerate(train_sets)
                             if j not in (i, i + 1)]

            # Collect training data
            """from utils import collect_training_data"""
            acts_centered, _, labels, polarities = (
                collect_training_data(dataset_names=cv_train_sets,
                                      train_set_sizes=train_set_sizes,
                                      model_family=model_family,
                                      model_size=model_size,
                                      model_type=model_type,
                                      layer=layer))

            # Fit model
            """from probes import learn_truth_directions
            t_g: General truth direction.
            t_p: Polarity sensitive truth direction.
            """
            t_g, t_p = learn_truth_directions(acts_centered=acts_centered,
                                              labels=labels,
                                              polarities=polarities)

            # fit LR for comparison
            """from sklearn.linear_model import LogisticRegression"""
            LR = LogisticRegression(penalty=None, fit_intercept=False)
            LR.fit(acts_centered.numpy(), labels.numpy())
            d_lr = torch.from_numpy(LR.coef_[0]).float()

            # Evaluate on held-out sets, assuming affirmative and
            # negated dataset on the same topic are at index i and i+1
            for j in range(2):
                dataset = train_sets[i + j]
                """from utils import DataManager"""
                dm = DataManager()
                dm.add_dataset(dataset_name=dataset, model_family=model_family,
                               model_size=model_size, model_type=model_type,
                               layer=layer, split=None, center=False,
                               scale=False, device='cpu')
                acts, labels = dm.get(dataset)
                """
                acts: torch.Size([1496, 4096])
                t_g.size(): torch.Size([4096])
                t_p: torch.Size([4096])
                d_lr: torch.Size([4096])
                """
                """from sklearn.metrics import roc_auc_score
                   Empirical distribution of activation vectors 
                corresponding to both affirmative and negated 
                statements projected onto t_G and t_A, respectively.
                   Interpretation as Projection:
                • Projection in Vector Space: The operation effectively 
                projects each 4096-dimensional vector in acts onto the 
                vector t_g. This results in a scalar that represents 
                how much of t_g is in the direction of each vector 
                in acts.
                • Geometrically: If you think of t_g as defining a line 
                through the origin in a 4096-dimensional space, then each 
                element of the resulting vector (after the 
                matrix-vector multiplication) represents the projection 
                of each row vector in acts onto this line.
                """
                auroc = roc_auc_score(labels.numpy(), (acts @ t_g).numpy())
                results['t_g'][dataset].append(auroc)
                auroc = roc_auc_score(labels.numpy(), (acts @ t_p).numpy())
                results['t_p'][dataset].append(auroc)
                auroc = roc_auc_score(labels.numpy(), (acts @ d_lr).numpy())
                results['d_{LR}'][dataset].append(auroc)

    """from utils import compute_statistics"""
    stat_results = compute_statistics(results=results)

    #### Figure
    # Create a custom colormap from red to yellow
    """from matplotlib.colors import LinearSegmentedColormap"""
    cmap = LinearSegmentedColormap.from_list('red_yellow',
                                             [(1, 0, 0), (1, 1, 0)],
                                             N=100)
    # Create three subplots side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(6, 6), ncols=3)
    for ax, key in zip((ax1, ax2, ax3), ('t_g', 't_p', 'd_{LR}')):
        grid = [[stat_results[key]['mean'][dataset]]
                for dataset in train_sets]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap)
        ax.set_aspect('auto')
        ax.set_aspect(0.6)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                ax.text(j, i, f'{val:.2f}', ha='center',
                        va='center', fontsize=13)

        ax.set_yticks(range(len(train_sets)))
        ax.set_xticks([])
        ax.set_title(f"${key}$", fontsize=14)

    ax1.set_yticklabels(train_sets, fontsize=10)
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    # Adjust the layout to make room for the colorbar
    plt.subplots_adjust(top=0.9, bottom=0.05,
                        left=0.25, right=0.85, wspace=0.4)

    # Add colorbar with a specified position
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle("AUROC", fontsize=15)
    plt.show()



### Define helper functions for plotting in the step3 and more below ###
def collect_affirm_neg_data(train_sets, train_set_sizes, model_family,
                            model_size, model_type, layer):
    dm_affirm, dm_neg = DataManager(), DataManager()
    for dataset_name in train_sets:
        split = min(train_set_sizes.values()) / train_set_sizes[dataset_name]
        if 'neg_' not in dataset_name:
            dm_affirm.add_dataset(dataset_name, model_family, model_size,
                                  model_type, layer, split=split,
                                  center=False, device='cpu')
        else:
            dm_neg.add_dataset(dataset_name, model_family, model_size,
                               model_type, layer, split=split,
                               center=False, device='cpu')
    return dm_affirm.get('train') + dm_neg.get('train')


def compute_t_affirm(acts_affirm, labels_affirm):
    LR = LogisticRegression(penalty=None, fit_intercept=True)
    LR.fit(acts_affirm.numpy(), labels_affirm.numpy())
    return LR.coef_[0] / np.linalg.norm(LR.coef_[0])


def compute_orthonormal_vectors(t_g, t_p):
    """
       Orthonormalise t_g and t_p, designed to perform the
    orthonormalization of two vectors, t_g and t_p.
    Orthonormalization is a process in linear algebra
    where a set of vectors is transformed into a set of
    orthonormal vectors, meaning the vectors are both orthogonal
    (perpendicular to each other) and normalized (each vector
    has a length or norm of 1). This function appears to employ
    the Gram-Schmidt process, a method for orthonormalizing a set
    of vectors in an inner product space, meaning that converting
    them into a set of orthogonal (perpendicular) vectors that are
    also normalized (unit length).
    """
    t_g_numpy = t_g.numpy()
    t_p_numpy = t_p.numpy()

    ## Calculate the projection of vector t_p onto vector t_g.
    projection = (np.dot(t_p_numpy, t_g_numpy) /
                  np.dot(t_g_numpy, t_g_numpy) * t_g_numpy)
    ## Normalize t_g to Create t_g_orthonormal
    t_g_orthonormal = t_g_numpy / np.linalg.norm(t_g_numpy)
    ## Subtract the projection of t_p onto t_g from t_p, resulting
    # in a vector perpendicular to t_g. Then normalizes the new
    # vector.
    t_p_orthonormal = ((t_p_numpy - projection) /
                       np.linalg.norm(t_p_numpy - projection))
    return t_g_orthonormal, t_p_orthonormal


def project_activations(acts, t_g, t_p):
    """
       acts.shape is torch.Size([984, 4096])
       t_g: <class 'numpy.ndarray'>, shape is (4096,)
       t_p: <class 'numpy.ndarray'>, shape is (4096,)
    """
    return t_g @ acts.numpy().T, t_p @ acts.numpy().T


def plot_vector(ax, vector, t_g_orthonormal,
                t_p_orthonormal, label, midpoint):
    # Normalize input vector
    vector_normalized = vector / np.linalg.norm(vector)

    # Compute vector_subspace
    vector_subspace = np.array([(np.dot(t_g_orthonormal,
                                        vector_normalized)),
                                (np.dot(t_p_orthonormal,
                                        vector_normalized))])

    vector_subspace = vector_subspace / np.linalg.norm(vector_subspace)

    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Compute scale based on axis limits
    axis_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    scale = 0.35 * axis_range  # Adjust this factor to change the relative size of the vector

    # Compute arrow head position
    arrow_head = np.array(midpoint) + scale * vector_subspace

    # Compute label offset based on axis limits
    label_offset = np.array([0.03 * (xlim[1] - xlim[0]),
                             0.03 * (ylim[1] - ylim[0])])

    # Adjust label position to avoid overlap with arrow
    label_position = arrow_head + label_offset * np.sign(vector_subspace)

    # Plot the vector
    ax.quiver(*midpoint, *(scale * vector_subspace),
              color='green', angles='xy', scale_units='xy', scale=1,
              width=0.03)

    # Add label
    ax.annotate(label, xy=label_position, fontsize=21,
                ha='center', va='center')


# Update the plot_scatter function to use the new plot_vector function
def plot_scatter(ax, proj_g, proj_p, labels, proj_g_other,
                 proj_p_other, labels_other,
                 title, plot_t_a=False, plot_t_g_t_p=False, **kwargs):
    label_to_color = {0: 'indigo', 1: 'orange'}
    label_to_marker = {0: 's', 1: '^'}

    for label in [0, 1]:
        idx = labels.numpy() == label
        ax.scatter(proj_g[idx], proj_p[idx], c=label_to_color[label],
                   marker=label_to_marker[label], alpha=0.5, s=5)
        idx_other = labels_other.numpy() == label
        if title == "Affirmative & Negated\nStatements":
            ax.scatter(proj_g_other[idx_other],
                       proj_p_other[idx_other],
                       c=label_to_color[label],
                       marker=label_to_marker[label], alpha=0.5, s=5)
        else:
            ax.scatter(proj_g_other[idx_other],
                       proj_p_other[idx_other],
                       c='grey', marker=label_to_marker[label],
                       alpha=0.1, s=5)

    # Compute midpoint based on current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    midpoint = (0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1]))

    if plot_t_a:
        plot_vector(ax=ax, vector=kwargs['t_affirm'],
                    t_g_orthonormal=kwargs['t_g_orthonormal'],
                    t_p_orthonormal=kwargs['t_p_orthonormal'],
                    label="$t_A$", midpoint=midpoint)
    if plot_t_g_t_p:
        plot_vector(ax=ax, vector=kwargs['t_g'],
                    t_g_orthonormal=kwargs['t_g_orthonormal'],
                    t_p_orthonormal=kwargs['t_p_orthonormal'],
                    label="$t_G$", midpoint=midpoint)
        plot_vector(ax=ax, vector=kwargs['t_p'],
                    t_g_orthonormal=kwargs['t_g_orthonormal'],
                    t_p_orthonormal=kwargs['t_p_orthonormal'],
                    label="$t_P$", midpoint=midpoint)

    ax.set_title(title, fontsize=19)
    # ax.set_yticks([])
    # ax.set_xticks([])

    # Update axis limits after plotting
    ax.autoscale()
    ax.set_aspect('equal')


def add_legend(ax):
    handles = [plt.scatter([], [], c='indigo',
                           marker='s', label='False'),
               plt.scatter([], [], c='orange',
                           marker='^', label='True')]
    ax.legend(handles=handles, fontsize=18)


def calculate_auroc(acts, labels, t):
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    proj = t @ acts.T
    auroc = roc_auc_score(labels, proj)
    return auroc


def plot_density(ax, acts, labels, t, xlabel):
    # Convert inputs to NumPy arrays if they're PyTorch tensors
    if isinstance(acts, torch.Tensor):
        acts = acts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    # Compute projections
    if t.ndim == 1:
        proj = t @ acts.T
    else:
        proj = t.reshape(1, -1) @ acts.T

    # Compute axis limits based on projected activations
    x_min, x_max = np.min(proj), np.max(proj)
    x_range = x_max - x_min
    x_padding = 0.1 * x_range  # Add 10% padding on each side
    xlim = (x_min - x_padding, x_max + x_padding)

    # Set x-axis limits
    ax.set_xlim(xlim)

    # Compute KDE for each label
    x_grid = np.linspace(xlim[0], xlim[1], 400)
    for label, color in zip([0, 1], ['indigo', 'orange']):
        data = proj[labels == label]
        kde = gaussian_kde(data)
        density = kde(x_grid)
        density /= np.trapz(density, x_grid)
        ax.plot(x_grid, density, color=color)

    # Plot scatter points
    y_scatter = np.ones(np.shape(proj)) * (-0.05)
    colors = ['indigo' if label == 0 else 'orange' for label in labels]
    ax.scatter(proj, y_scatter, c=colors, alpha=0.3, s=10)

    # Set y-axis limits to accommodate both KDE and scatter points
    y_max = ax.get_ylim()[1]
    # Extend y-axis slightly above the maximum KDE value
    ax.set_ylim(-0.1, y_max * 1.1)

    # Calculate AUROC
    auroc = calculate_auroc(acts, labels, t)

    # Display AUROC in the top left corner
    ax.text(0.05, 0.95, f'AUROC: {auroc:.2f}',
            transform=ax.transAxes,
            verticalalignment='top', fontsize=14,
            bbox=dict(facecolor='white', alpha=0.7))

    # Set labels and remove ticks
    ax.set_ylabel('Frequency', fontsize=19)
    ax.set_xlabel(xlabel, fontsize=19)
    ax.set_yticks([])
    ax.set_xticks([])

    # Add a light grid
    ax.grid(True, linestyle='--', alpha=0.3)


### Define helper functions for plotting in the step3 above ###
def run_step3(train_sets, train_set_sizes, model_family,
              model_size, model_type, layer):
    """
       Activation vectors projected onto 2d truth subspace.
    """
    # Compute t_g and t_p using all data
    """from utils import collect_training_data"""
    acts_centered, _, labels, polarities = (
        collect_training_data(dataset_names=train_sets,
                              train_set_sizes=train_set_sizes,
                              model_family=model_family,
                              model_size=model_size,
                              model_type=model_type,
                              layer=layer))

    """from probes import learn_truth_directions
    t_g: General truth direction.
    t_p: Polarity sensitive truth direction.
    """
    t_g, t_p = learn_truth_directions(acts_centered=acts_centered,
                                      labels=labels,
                                      polarities=polarities)

    ### Figure 1 below
    fig = plt.figure(figsize=(11.5, 11))
    axes = [
        fig.add_axes([0.4, 0.4, 0.26, 0.26]),  # ax1: top center
        fig.add_axes([0.7, 0.4, 0.26, 0.26]),  # ax2: top right
        fig.add_axes([0.1, 0.4, 0.26, 0.26]),  # ax3: top left
        fig.add_axes([0.58, 0.1, 0.25, 0.25]),  # ax4: bottom right
        fig.add_axes([0.23, 0.1, 0.25, 0.25])  # ax5: bottom left
    ]

    """from truth_directions_main import collect_affirm_neg_data
       Collect activations and labels of affirmative and negated
    statements separately.
    """
    acts_affirm, labels_affirm, acts_neg, labels_neg = (
        collect_affirm_neg_data(train_sets=train_sets,
                                train_set_sizes=train_set_sizes,
                                model_family=model_family,
                                model_size=model_size,
                                model_type=model_type, layer=layer))

    """from truth_directions_main import compute_t_affirm
       Compute t_affirm via logistical regression 
    from acts_affirm.numpy() and labels_affirm.numpy().
       t_affirm: <class 'numpy.ndarray'>, shape is (4096,)
    """
    t_affirm = compute_t_affirm(acts_affirm=acts_affirm,
                                labels_affirm=labels_affirm)

    """from truth_directions_main import compute_orthonormal_vectors
       Employ the Gram-Schmidt processOrthonormalise t_g and t_p. As 
    mentioned in the introduction, the authors demonstrate the existence 
    of two truth directions in the activation space: the general truth 
    direction t_G and the polarity-sensitive truth direction t_P. In 
    Figure 1 the authors visualise the projections of the activations 
    a_ij onto the 2D subspace spanned by our estimates of the vectors 
    t_G and t_P. In this visualization of the subspace, we choose the 
    orthonormalized versions of tG and tP as its basis. We discuss the 
    reasons for this choice of basis for the 2D subspace in Appendix B.
       Difference Between the New and Original Vectors
    t_g_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
    t_p_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
    1. Orthogonality:
    • Original vectors t_g and t_p may not be orthogonal.
    • New vectors t_g_orthonormal and t_p_orthonormal are 
    explicitly made orthogonal.
    2. Normalization:
    • The original vectors t_g and t_p can have any length.
    • The new vectors t_g_orthonormal and t_p_orthonormal 
    are normalized to unit length.
    3. Direction:
    • t_g_orthonormal has the same direction as t_g but is 
    scaled to unit length.
    • t_p_orthonormal is constructed by removing the projection of t_p 
    onto t_g. Therefore, t_p_orthonormal lies in the same plane as t_g 
    and t_p, but its direction differs from t_p .
    4. Dependence:
    • t_g_orthonormal and t_p_orthonormal still span the same subspace 
    as t_g and t_p.
    • The new vectors form an orthonormal basis for the same subspace 
    as the original vectors.
    """
    t_g_orthonormal, t_p_orthonormal = (
        compute_orthonormal_vectors(t_g=t_g, t_p=t_p))

    ## Project activations
    """from truth_directions_main import project_activations"""
    """   
       acts_affirm.shape is torch.Size([984, 4096])
       t_g_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       t_p_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       proj_g_affirm: <class 'numpy.ndarray'>, shape is (984,)
       proj_p_affirm: <class 'numpy.ndarray'>, shape is (984,)
    """
    proj_g_affirm, proj_p_affirm = project_activations(acts=acts_affirm,
                                                       t_g=t_g_orthonormal,
                                                       t_p=t_p_orthonormal)
    """   
       acts_neg.shape is torch.Size([984, 4096])
       t_g_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       t_p_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       proj_g_neg: <class 'numpy.ndarray'>, shape is (984,)
       proj_p_neg: <class 'numpy.ndarray'>, shape is (984,)
    """
    proj_g_neg, proj_p_neg = project_activations(acts=acts_neg,
                                                 t_g=t_g_orthonormal,
                                                 t_p=t_p_orthonormal)

    ## Plot scatter plots
    """from truth_directions_main import plot_scatter
       t_g_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       t_p_orthonormal: <class 'numpy.ndarray'>, shape is (4096,)
       proj_g_affirm: <class 'numpy.ndarray'>, shape is (984,)
       proj_p_affirm: <class 'numpy.ndarray'>, shape is (984,)
       proj_g_neg: <class 'numpy.ndarray'>, shape is (984,)
       proj_p_neg: <class 'numpy.ndarray'>, shape is (984,)
    """
    plot_scatter(ax=axes[0], proj_g=proj_g_affirm, proj_p=proj_p_affirm,
                 labels=labels_affirm,
                 proj_g_other=proj_g_neg, proj_p_other=proj_p_neg,
                 labels_other=labels_neg,
                 title='Affirmative Statements', plot_t_a=True,
                 t_affirm=t_affirm, t_g_orthonormal=t_g_orthonormal,
                 t_p_orthonormal=t_p_orthonormal)
    plot_scatter(ax=axes[1], proj_g=proj_g_neg, proj_p=proj_p_neg,
                 labels=labels_neg,
                 proj_g_other=proj_g_affirm, proj_p_other=proj_p_affirm,
                 labels_other=labels_affirm,
                 title='Negated Statements')
    plot_scatter(ax=axes[2], proj_g=proj_g_affirm, proj_p=proj_p_affirm,
                 labels=labels_affirm,
                 proj_g_other=proj_g_neg, proj_p_other=proj_p_neg,
                 labels_other=labels_neg,
                 title='Affirmative & Negated\nStatements',
                 plot_t_g_t_p=True,
                 t_g=t_g, t_p=t_p, t_g_orthonormal=t_g_orthonormal,
                 t_p_orthonormal=t_p_orthonormal)

    ## Add legend
    """from truth_directions_main import add_legend"""
    add_legend(ax=axes[2])

    ## Plot density plots
    acts = torch.cat((acts_affirm, acts_neg), dim=0)
    labels = torch.cat((labels_affirm, labels_neg))
    """from truth_directions_main import plot_density"""
    plot_density(ax=axes[3], acts=acts, labels=labels,
                 t=t_affirm, xlabel='$a^T t_A$')
    plot_density(ax=axes[4], acts=acts, labels=labels,
                 t=t_g, xlabel='$a^T t_G$')
    plt.show()
    ### Figure 1 above


    #### New
    ### Activation vectors projected onto t_G and t_P (reduced version of figure 1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    """from truth_directions_main import collect_affirm_neg_data"""
    acts_affirm, labels_affirm, acts_neg, labels_neg = (
        collect_affirm_neg_data(train_sets=train_sets,
                                train_set_sizes=train_set_sizes,
                                model_family=model_family,
                                model_size=model_size,
                                model_type=model_type, layer=layer))

    for i, (acts, labels) in enumerate([(acts_affirm, labels_affirm),
                                        (acts_neg, labels_neg)]):
        prod_g, prod_p = project_activations(acts=acts, t_g=t_g, t_p=t_p)
        ax = axes[i]
        if i == 0:
            ax.set_xlabel('$a_{ij}^T t_G$', fontsize=19)
            ax.set_ylabel('$a_{ij}^T t_P$', fontsize=19)
            ax.set_title('Affirmative Statements', fontsize=19)
        else:
            ax.set_title('Negated Statements', fontsize=19)

        colors = ['red' if label == 0 else 'blue' for label in labels]
        ax.scatter(prod_g, prod_p, c=colors, alpha=0.5, s=5)

    # Add the legend to the last subplot
    handles = [plt.scatter([], [], c='red', label='False'),
               plt.scatter([], [], c='blue', label='True')]
    axes[1].legend(handles=handles, fontsize=19)

    fig.suptitle('Projection of activations on $t_G$ and $t_P$', fontsize=19)
    plt.show()


    #### New Figure 11-a: LLaMA2-13B: Left (a): Activations a_ij
    # projected onto t_G and t_P.

    """
       Projection of other datasets onto t_G and t_P
    - larger_than and smaller_than are shown as examples.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    acts_affirm, labels_affirm, acts_neg, labels_neg = (
        collect_affirm_neg_data(train_sets, train_set_sizes,
                                model_family, model_size, model_type,
                                layer))
    # Project activations on t_g and t_p
    proj_g_affirm, proj_p_affirm = project_activations(acts_affirm, t_g, t_p)
    proj_g_neg, proj_p_neg = project_activations(acts_neg, t_g, t_p)

    # Define colors and markers for each label
    label_to_color = {0: 'indigo', 1: 'orange'}
    label_to_marker = {0: 's', 1: '^'}  # s for square, ^ for triangle

    for i, dataset_name in enumerate(['larger_than', 'smaller_than']):
        ax = axes[i]
        ax.set_title(dataset_name, fontsize=19)
        for label in [0, 1]:
            idx = labels_affirm.numpy() == label
            ax.scatter(proj_g_affirm[idx], proj_p_affirm[idx], c='grey',
                       marker=label_to_marker[label], alpha=0.3, s=5)
            idx = labels_neg.numpy() == label
            ax.scatter(proj_g_neg[idx], proj_p_neg[idx], c='grey',
                       marker=label_to_marker[label], alpha=0.3, s=5)

        """from utils import DataManager"""
        dm = DataManager()
        dm.add_dataset(dataset_name=dataset_name, model_family=model_family,
                       model_size=model_size, model_type=model_type,
                       layer=layer, split=None, center=False, scale=False,
                       device='cpu')
        acts, labels = dm.data[dataset_name]
        prod_g, prod_p = project_activations(acts, t_g, t_p)
        for label in [0, 1]:
            idx = labels.numpy() == label
            ax.scatter(prod_g[idx], prod_p[idx], c=label_to_color[label],
                       marker=label_to_marker[label], alpha=0.9, s=15)
            if i == 0:
                ax.set_xlabel('$a^T t_G$', fontsize=19)
                ax.set_ylabel('$a^T t_P$', fontsize=19)

    add_legend(axes[0])
    fig.suptitle('Projection of activations on $t_G$ and $t_P$', fontsize=19)
    plt.show()

    return t_g, t_p


def compute_subspace_angle(A, B):
    """
       Computes the principal angles between
    two subspaces represented by the columns
    of matrices A and B. This is typically used
    in numerical linear algebra and machine learning
    to quantify the similarity or alignment
    between two subspaces.
    """

    """
       Normalize columns of A and B.
    Purpose: Normalization is crucial because it 
    allows the subsequent computations to measure 
    angles in a consistent manner, ensuring that 
    only the directions of the vectors influence 
    the result, not their magnitudes.
    """
    A = A / torch.linalg.norm(A, dim=0)
    B = B / torch.linalg.norm(B, dim=0)

    # Compute SVD of A^T * B
    U, S, Vt = torch.linalg.svd(A.T @ B)


    """
       Compute principal angles
    • Clamping: torch.clamp(S, -1, 1) ensures that 
    the values in S (which should theoretically be 
    between -1 and 1 as they are cosines of angles) 
    are indeed within this range. This is important 
    to avoid numerical errors that might occur due 
    to floating-point precision issues, leading to 
    input values slightly outside this range for arccos.
    • arccos: torch.arccos() computes the arc cosine 
    of each value in S, yielding the principal angles 
    in radians. These angles represent the angles between 
    the corresponding principal vectors of the subspaces 
    spanned by A and B.
    """
    angles = torch.arccos(torch.clamp(S, -1, 1))

    return angles


def run_step4(model_family, model_size, model_type, layer, t_g, t_p):
    """
       Dimensionality of Truth in Section 4 of the paper.
    Are there more than two truth dimensions? Fraction of truth
    related variance in activations explained by Principal
    Components.
    """
    # Define the four different statement types and corresponding datasets
    statement_types = ["affirmative",
                       "affirmative, negated",
                       "affirmative, negated, conjunctions",
                       "affirmative, affirmative German",
                       "affirmative, affirmative German,\nnegated, negated German",
                       "affirmative, negated,\nconjunctions, disjunctions"]
    datasets_pca_options = {
        "affirmative": ['cities', 'sp_en_trans', 'inventors',
                        'animal_class', 'element_symb', 'facts'],  # top left

        "affirmative, negated": ['cities', 'sp_en_trans', 'inventors',
                                 'animal_class', 'element_symb', 'facts',
                                 'neg_cities', 'neg_sp_en_trans',
                                 'neg_inventors', 'neg_animal_class',
                                 'neg_element_symb', 'neg_facts'
                                 ],  # top middle

        "affirmative, negated, conjunctions":
            ['cities', 'sp_en_trans', 'inventors',
             'animal_class', 'element_symb', 'facts',
             'neg_cities', 'neg_sp_en_trans',
             'neg_inventors', 'neg_animal_class',
             'neg_element_symb', 'neg_facts',
             'cities_conj', 'sp_en_trans_conj', 'inventors_conj',
             'animal_class_conj', 'element_symb_conj', 'facts_conj'
             ],  # top right

        "affirmative, affirmative German":
            ['cities', 'sp_en_trans', 'inventors',
             'animal_class', 'element_symb', 'facts',
             'cities_de', 'sp_en_trans_de',
             'inventors_de', 'animal_class_de',
             'element_symb_de', 'facts_de',
             ],  # bottom left

        "affirmative, affirmative German,\nnegated, negated German":
            ['cities', 'sp_en_trans', 'inventors', 'animal_class',
             'element_symb', 'facts',
             'cities_de', 'sp_en_trans_de', 'inventors_de', 'animal_class_de',
             'element_symb_de', 'facts_de',
             'neg_cities', 'neg_sp_en_trans', 'neg_inventors',
             'neg_animal_class', 'neg_element_symb', 'neg_facts',
             'neg_cities_de', 'neg_sp_en_trans_de', 'neg_inventors_de',
             'neg_animal_class_de', 'neg_element_symb_de', 'neg_facts_de'
             ],  # bottom middle

        "affirmative, negated,\nconjunctions, disjunctions":
            ['cities', 'sp_en_trans', 'inventors', 'animal_class',
             'element_symb', 'facts',
             'neg_cities', 'neg_sp_en_trans', 'neg_inventors', 'neg_animal_class',
             'neg_element_symb', 'neg_facts',
             'cities_conj', 'sp_en_trans_conj', 'inventors_conj',
             'animal_class_conj', 'element_symb_conj', 'facts_conj',
             'cities_disj', 'sp_en_trans_disj', 'inventors_disj',
             'animal_class_disj', 'element_symb_disj', 'facts_disj'
             ]  # bottom right
    }


    # Create the 2x2 plot
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    for i, statement_type in enumerate(statement_types):
        directions = []
        datasets_pca = datasets_pca_options[statement_type]
        for train_set in datasets_pca:
            """from utils import DataManager"""
            dm = DataManager()
            dm.add_dataset(dataset_name=train_set, model_family=model_family,
                           model_size=model_size, model_type=model_type,
                           layer=layer, split=1.0, center=True,
                           scale=False, device='cpu')
            train_acts, train_labels = dm.get('train')
            true_acts = train_acts[train_labels.to(bool)]
            # true_acts: torch.Size([748, 4096]), ~u_i^+ in Eq.(6)
            false_acts = train_acts[~train_labels.to(bool)]  # ~: negate
            # false_acts: torch.Size([748, 4096]), ~u_i^- in Eq.(6)
            directions.append(torch.mean(true_acts, dim=0))  # torch.Size([4096])
            directions.append(torch.mean(false_acts, dim=0))  # torch.Size([4096])

        mean_acts = np.array([direction.numpy() for direction in directions])
        """from sklearn.decomposition import PCA
           How PCA works briefly:
        1.Standardizing the Data: PCA often starts by standardizing 
        the data (mean = 0 and variance = 1).
        2.Covariance Matrix Computation: It computes the covariance matrix 
        of the data to understand how variables are linearly related.
        3.Eigendecomposition: PCA involves calculating the eigenvalues and 
        eigenvectors of this covariance matrix. The eigenvectors determine 
        the directions of the new feature space, and the eigenvalues determine 
        their magnitude. In essence, the eigenvalues explain the variance of 
        the data along the new feature axes.
        4.Sorting Eigenvalues: The eigenvalues are sorted in descending order 
        to rank the corresponding eigenvectors in order of importance.
           explained_variance_ratio_ in Action:
        • n_components=10: In your PCA model, you specify n_components=10, 
        which means PCA will reduce the dimensionality of the original data 
        to 10 principal components.
        • Variance Explained: The explained_variance_ratio_ attribute will 
        then contain 10 values, each indicating the percentage of the dataset’s 
        total variance that is explained by each of these 10 components.
           Practical Implication:
        • Interpretation: Suppose the first element of explained_variance_ratio_ 
        is 0.5 (or 50%). This means that the first principal component alone 
        accounts for 50% of the variance in the entire dataset. Similarly, if 
        the second value is 0.25, the second component explains an additional 
        25% of the variance, and so on.
        • Sum of Ratios: The sum of all the ratios can give you an idea of the 
        total variance explained by the selected components. For instance, if 
        all 10 components together explain 95% of the total variance, you might 
        conclude that these components effectively represent most of the 
        underlying structure of the data.
        """
        pca = PCA(n_components=10)
        pca.fit(mean_acts)
        axs[i].scatter(np.arange(1, 11, 1),
                       pca.explained_variance_ratio_,
                       s=90)
        axs[i].set_title(f'{statement_type}', fontsize=26)
        if i == 0 or i == 3:
            axs[i].set_ylabel('Explained variance', fontsize=27)
        if i == 3 or i == 4 or i == 5:
            axs[i].set_xlabel('PC index', fontsize=26)
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        axs[i].grid(True)
        if statement_type == "affirmative, negated":
            """
            from probes import learn_truth_directions
            t_g: General truth direction.
            t_p: Polarity sensitive truth direction.
            t_g, t_p = learn_truth_directions(
               acts_centered=acts_centered,
               labels=labels, polarities=polarities)
            from train_sets = ["cities", "neg_cities", 
            "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors", 
            "animal_class", "neg_animal_class", "element_symb", "neg_element_symb", 
            "facts", "neg_facts"]
            """
            ## Compute subspace angle
            A = torch.stack([t_g, t_p], dim=1)
            B = torch.stack([torch.tensor(pca.components_[0, :]),
                             torch.tensor(pca.components_[1, :])], dim=1)
            """from truth_directions_main import compute_subspace_angle"""
            angles = compute_subspace_angle(A=A, B=B)
            print(f"Principal angles between subspaces (in radians): "
                  f"{angles}")
            print(f"Principal angles between subspaces (in degrees): "
                  f"{torch.rad2deg(angles)}")
            print("Cosine similarity between t_G and first PC: " + str(
                torch.tensor(pca.components_[0, :]) /
                torch.linalg.norm(torch.tensor(pca.components_[0, :]))
                @ t_g / np.linalg.norm(t_g)))
            print("Cosine similarity between t_P and second PC: " + str(
                torch.tensor(pca.components_[1, :]) /
                torch.linalg.norm(torch.tensor(pca.components_[1, :]))
                @ t_p / np.linalg.norm(t_p)))

    fig.suptitle('Fraction of variance in centered and averaged\n '
                 'activations explained by PCs', fontsize=28)
    plt.tight_layout()
    plt.show()


def run_step5(train_sets, train_set_sizes, model_family,
              model_size, model_type, layer):
    """
       Generalisation accuracies of truth directions
    trained on different data.
    """
    train_sets_subset = [
        ['cities'], ['cities', 'neg_cities'],
        ['cities', 'neg_cities', 'cities_conj'],
        ['cities', 'neg_cities', 'cities_conj', 'cities_disj']
    ]  # The x-axis of the Figure 5
    val_sets_subset = \
        ['cities', 'neg_cities', 'facts',
         'neg_facts', 'facts_conj', 'facts_disj'
         ]  # The y-axis of the Figure 5

    num_runs = 10
    project_options = [None, 't_G_t_P']

    ## Helper function to create unique keys for training sets
    def get_train_set_key(train_set):
        return '_'.join(train_set)

    ## Initialize dictionaries to store accuracies for each projection option
    all_aurocs_options = {
        proj: {get_train_set_key(train_set):
                   {val_set: [] for val_set in val_sets_subset}
               for train_set in train_sets_subset}
        for proj in project_options}

    for project_out in project_options:
        all_aurocs = all_aurocs_options[project_out]

        for run in range(num_runs):
            # Compute t_g and t_p using all data
            """from utils import collect_training_data"""
            acts_centered, _, labels, polarities = collect_training_data(
                dataset_names=train_sets, train_set_sizes=train_set_sizes,
                model_family=model_family, model_size=model_size,
                model_type=model_type,
                layer=layer)
            """from probes import learn_truth_directions"""
            t_g, t_p = learn_truth_directions(acts_centered=acts_centered,
                                              labels=labels,
                                              polarities=polarities)
            # orthonormalize t_g and t_p
            t_G_orthonormal, t_P_orthonormal = (
                compute_orthonormal_vectors(t_g=t_g, t_p=t_p))

            for train_set in train_sets_subset:
                train_set_key = get_train_set_key(train_set)

                # set up data
                """from utils import DataManager"""
                dm = DataManager()
                for subset in train_set:
                    dm.add_dataset(dataset_name=subset,
                                   model_family=model_family,
                                   model_size=model_size,
                                   model_type=model_type,
                                   layer=layer, split=0.8, center=True,
                                   scale=False, device='cpu')
                train_acts, train_labels = dm.get('train')
                if project_out is None:
                    pass
                elif project_out == 't_G_t_P':
                    """
                       Project activations onto the orthogonal 
                    complement of Span(t_G, t_P).
                    • The updated train_acts has been purged of any 
                    variance that lies along the two specified directions 
                    (t_G_orthonormal and t_P_orthonormal). This is useful 
                    in scenarios where these directions might represent 
                    noise, redundant information, or undesirable data 
                    characteristics.
                    • This kind of operation is typical in advanced data 
                    preprocessing, feature orthogonalization, and is critical 
                    in preparing data for some types of statistical analyses 
                    or machine learning algorithms where independence of 
                    features is required.
                       The code effectively creates a new representation of 
                    the data in train_acts that is cleaner in the sense that 
                    it is now orthogonal to the specified directions, potentially 
                    leading to better performance in downstream tasks such as 
                    clustering, dimensionality reduction, or predictive modeling.
                       
                       train_acts - (train_acts @ t_G_orthonormal)[:, None] 
                    * t_G_orthonormal: This operation removes the component of 
                    each vector in train_acts that lies along t_G_orthonormal. 
                    After this step, the train_acts vectors are orthogonal to 
                    t_G_orthonormal.
                    """
                    train_acts = (train_acts -
                                  (train_acts @ t_G_orthonormal)
                                  [:, None] * t_G_orthonormal
                                  -
                                  (train_acts @ t_P_orthonormal)
                                  [:, None] * t_P_orthonormal)

                """
                  We compute each t using the supervised learning approach 
                from Section 3, with all polarities pi set to zero to learn 
                a single truth direction.
                """
                polarities = torch.zeros((train_labels.shape)[0])
                # learn t_G
                """from probes import learn_truth_directions"""
                t_g_trained, _ = learn_truth_directions(acts_centered=train_acts,
                                                        labels=train_labels,
                                                        polarities=polarities)

                # compute auroc of a^T t_G on validation sets
                for val_set in val_sets_subset:
                    if val_set in train_set:
                        acts, labels = dm.get('val')
                    else:
                        dm.add_dataset(dataset_name=val_set,
                                       model_family=model_family,
                                       model_size=model_size,
                                       model_type=model_type,
                                       layer=layer, split=None,
                                       center=False,
                                       scale=False, device='cpu')
                        acts, labels = dm.data[val_set]

                    proj_g = acts @ t_g_trained
                    """from sklearn.metrics import roc_auc_score"""
                    auroc = roc_auc_score(labels.numpy(), proj_g.numpy())
                    all_aurocs[train_set_key][val_set].append(auroc)

        # Calculate mean and standard deviation for each training-validation
        # set combination
        mean_aurocs = {
            train_set:
                {
                    val_set: np.mean(accs)
                    for val_set, accs in val_sets.items()
                }
            for train_set, val_sets in all_aurocs.items()
        }
        std_aurocs = {
            train_set: {
                val_set: np.std(accs)
                for val_set, accs in val_sets.items()
            }
            for train_set, val_sets in all_aurocs.items()
        }

        all_aurocs_options[project_out] = {'mean': mean_aurocs,
                                           'std': std_aurocs}
        print(mean_aurocs, std_aurocs)
        print("=>=> project_out: {}".format(project_out))
        print("=>=> all_aurocs_options[project_out]: {}"
              .format(all_aurocs_options[project_out]))


    ## Plotting the results
    fig, axes = plt.subplots(figsize=(14, 11), nrows=1,
                             ncols=2, sharey=True)
    # Titles for the x and y axes
    titles_val = ['cities', 'neg_cities', 'facts', 'neg_facts',
                  'facts_conj', 'facts_disj']
    titles_train = ['cities', '+ neg_cities', '+ cities_conj', '+ cities_disj']

    # Create a custom colormap from red to yellow
    colors = [(1, 0, 0), (1, 1, 0)]  # Red to Yellow
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'red_yellow'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    for idx, project_out in enumerate(project_options):
        mean_aurocs = all_aurocs_options[project_out]['mean']
        std_aurocs = all_aurocs_options[project_out]['std']

        # Prepare the grid for mean accuracies
        grid = np.zeros((len(train_sets_subset), len(val_sets_subset)))

        # Populate the grid with mean accuracies
        for i, train_set in enumerate(train_sets_subset):
            train_set_key = get_train_set_key(train_set)
            for j, val_set in enumerate(val_sets_subset):
                grid[i, j] = mean_aurocs[train_set_key][val_set]

        # Plot the grid
        im = axes[idx].imshow(grid.T, vmin=0, vmax=1, cmap=cmap, aspect='auto')

        # Annotate each cell with the mean accuracy and standard deviation
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                mean_auroc = grid[i][j]
                std_auroc = std_aurocs[
                    get_train_set_key(train_sets_subset[i])
                ][val_sets_subset[j]]
                axes[idx].text(i, j, f'{mean_auroc:.2f}',
                               ha='center', va='center',
                               fontsize=16)  # ±{std_auroc:.2f}

        # Titles for the x and y axes
        axes[idx].set_yticks(range(len(val_sets_subset)))
        axes[idx].set_xticks(range(len(train_sets_subset)))
        axes[idx].set_yticklabels([val_title
                                   for val_title in titles_val], fontsize=16)
        axes[idx].set_xticklabels([train_title
                                   for train_title in titles_train],
                                  rotation=15, ha='right', fontsize=16)

        # Set title and labels for the subplot
        if idx == 0:
            axes[idx].set_title(f'Projected out: None', fontsize=20)
        if idx == 1:
            axes[idx].set_title(f'Projected out: $t_G$ and $t_P$', fontsize=20)
        if idx == 0:
            axes[idx].set_ylabel('Test Set', fontsize=20)
            axes[idx].set_xlabel('Train Set "cities"', fontsize=20)

    # Add colorbar to the last subplot
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    fig.suptitle('AUROC for Projections $a^T t$', fontsize=20, x=0.42)

    # Show the plot
    plt.show()


def run_step6(model_family, model_size, model_type, layer):
    train_sets_subset = [['cities'], ['neg_cities'],
                         ['cities', 'neg_cities'],
                         ['cities_conj'], ['cities_disj']]

    val_sets_subset = ['cities', 'neg_cities',
                       'facts', 'neg_facts',
                       'facts_conj', 'facts_disj']

    num_runs = 10

    # Helper function to create unique keys for training sets
    def get_train_set_key(train_set):
        return '_'.join(train_set)

    # Initialize dictionaries to store accuracies for each projection option
    all_aurocs = {get_train_set_key(train_set):
                      {val_set: []
                       for val_set in val_sets_subset}
                  for train_set in
                  train_sets_subset}
    for run in range(num_runs):
        for train_set in train_sets_subset:
            train_set_key = get_train_set_key(train_set)

            # set up data
            """from utils import DataManager"""
            dm = DataManager()
            for subset in train_set:
                dm.add_dataset(dataset_name=subset, model_family=model_family,
                               model_size=model_size, model_type=model_type,
                               layer=layer, split=0.8, center=True,
                               scale=False, device='cpu')
            train_acts, train_labels = dm.get('train')
            polarities = torch.zeros((train_labels.shape)[0])
            # learn t_G
            """from probes import learn_truth_directions"""
            t_g_trained, _ = learn_truth_directions(acts_centered=train_acts,
                                                    labels=train_labels,
                                                    polarities=polarities)

            # compute auroc of a^T t_G on validation sets
            for val_set in val_sets_subset:
                if val_set in train_set:
                    acts, labels = dm.get('val')
                else:
                    dm.add_dataset(dataset_name=val_set, model_family=model_family,
                                   model_size=model_size, model_type=model_type,
                                   layer=layer, split=None, center=False,
                                   scale=False, device='cpu')
                    acts, labels = dm.data[val_set]

                proj_g = acts @ t_g_trained
                """from sklearn.metrics import roc_auc_score"""
                auroc = roc_auc_score(labels.numpy(), proj_g.numpy())
                all_aurocs[train_set_key][val_set].append(auroc)

    # Calculate mean and standard deviation for each
    # training-validation set combination
    mean_aurocs = {train_set: {val_set: np.mean(accs)
                               for val_set, accs in val_sets.items()}
                   for train_set, val_sets in
                   all_aurocs.items()}
    std_aurocs = {train_set: {val_set: np.std(accs)
                              for val_set, accs in val_sets.items()}
                  for train_set, val_sets in
                  all_aurocs.items()}

    all_aurocs = {'mean': mean_aurocs, 'std': std_aurocs}
    print(mean_aurocs, std_aurocs)


    ### Plotting the results
    fig, ax = plt.subplots(figsize=(10, 11), nrows=1, ncols=1, sharey=True)
    # Titles for the x and y axes
    titles_val = ['cities', 'neg_cities', 'facts', 'neg_facts',
                  'facts_conj', 'facts_disj']
    titles_train = ['cities', 'neg_cities', 'cities+neg_cities',
                    'cities_conj', 'cities_disj']

    # Create a custom colormap from red to yellow
    colors = [(1, 0, 0), (1, 1, 0)]  # Red to Yellow
    n_bins = 100  # Discretize the interpolation into bins
    cmap_name = 'red_yellow'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    mean_aurocs = all_aurocs['mean']
    std_aurocs = all_aurocs['std']

    # Prepare the grid for mean accuracies
    grid = np.zeros((len(train_sets_subset), len(val_sets_subset)))

    # Populate the grid with mean accuracies
    for i, train_set in enumerate(train_sets_subset):
        train_set_key = get_train_set_key(train_set)
        for j, val_set in enumerate(val_sets_subset):
            grid[i, j] = mean_aurocs[train_set_key][val_set]

    # Plot the grid
    im = ax.imshow(grid.T, vmin=0, vmax=1, cmap=cmap, aspect='auto')

    # Annotate each cell with the mean accuracy and standard deviation
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            mean_auroc = grid[i][j]
            std_auroc = std_aurocs[
                get_train_set_key(train_sets_subset[i])
            ][val_sets_subset[j]]
            ax.text(i, j, f'{mean_auroc:.2f}', ha='center',
                    va='center', fontsize=16)  # ±{std_auroc:.2f}

        # Titles for the x and y axes
        ax.set_yticks(range(len(val_sets_subset)))
        ax.set_xticks(range(len(train_sets_subset)))
        ax.set_yticklabels([val_title for val_title in titles_val],
                           fontsize=15)
        ax.set_xticklabels([train_title for train_title in titles_train],
                           rotation=15, ha='right', fontsize=15)

        ax.set_ylabel('Test Set', fontsize=20)
        ax.set_xlabel('Train Set', fontsize=20)

    # Add colorbar to the last subplot
    cbar = fig.colorbar(im, shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    fig.suptitle('AUROC for Projections $a^T t$', fontsize=20, x=0.42)

    # Show the plot
    plt.show()

    data = np.zeros((100, 2))
    # zeroth feature is perfectly predictive of label
    data[0:50, 0] = 1.0
    # first feature is correlated with the correct label
    # but not perfectly, 20 correct, 5 incorrect
    data[0:20, 1] = 1.0
    data[50:55, 1] = 1.0
    labels = np.concatenate((np.ones(50), np.zeros(50)))
    # which method can disentangle feature 0 from feature 1?
    # mass mean
    d_mm = (np.mean(data[labels == 1.0], axis=0)
            - np.mean(data[labels == 0.0], axis=0))
    print(d_mm)
    # LR
    LR = LogisticRegression(penalty=None, fit_intercept=True)
    LR.fit(data, labels)
    d_LR = LR.coef_
    print(LR.intercept_)
    print(d_LR)


def main():
    # Hyperparameters
    """
       'Llama3', 'Llama2', 'Gemma', 'Gemma2' or 'Mistral'
    """
    model_family = 'Llama3'
    model_size = '8B'
    model_type = 'chat'  # options are 'chat' or 'base'
    model_type_list = ['chat', 'chat_pruned0.2', 'chat_pruned0.3',
                       'chat_pruned0.4', 'chat_pruned0.5',
                       'chat_pruned0.6', 'chat_pruned0.65',
                       'chat_pruned0.7']

    run_step = {"step1": True, "step2": False, "step3": False,
                "step4": False, "step5": False, "step6": False}

    # Define datasets used for training
    # the ordering [affirmative_dataset1, negated_dataset1,
    # affirmative_dataset2, negated_dataset2, ...]
    # is required by some functions.
    train_sets = ["cities", "neg_cities",
                  "sp_en_trans", "neg_sp_en_trans",
                  "inventors", "neg_inventors",
                  "animal_class", "neg_animal_class",
                  "element_symb", "neg_element_symb",
                  "facts", "neg_facts"]

    # Get size of each training dataset to include an equal number
    # of statements from each topic in training data
    """from utils import dataset_sizes"""
    train_set_sizes = dataset_sizes(datasets=train_sets)
    print("\n=>=> train_set_sizes is {}\n".format(train_set_sizes))

    if run_step["step1"]:
        """
           Plot the Figure 2: Ratio of the between-class variance 
        and within-class variance of activations corresponding to 
        true and false statements, across residual stream layers, 
        averaged over all dimensions of the respective layer.
        """
        print("\n=>=> You are running the step 1...\n")
        """
        for model_type in model_type_list:
            print("\n")
            run_step1(model_family=model_family,
                      model_size=model_size,
                      model_type=model_type)
            print("\n")
        """

        layers = np.arange(1, 27, 1)

        datasets_separation = ['cities', 'neg_cities',
                               'sp_en_trans', 'neg_sp_en_trans']
        for dataset in datasets_separation:
            array_path = "./figures/figure2/" + dataset

            """
            model_type_dict = {
                'chat': None,  'chat_pruned0.3': None,
                'chat_pruned0.4': None,
                'chat_pruned0.5': None, 'chat_pruned0.6': None,
                'chat_pruned0.65': None, 'chat_pruned0.7': None
            }
            """
            model_type_dict = {
                'chat': None,  'chat_pruned0.4': None,
                'chat_pruned0.5': None, 'chat_pruned0.6': None,
                'chat_pruned0.65': None
            }

            # model_type_dict = {'chat': None, 'chat_pruned0.5': None}
            for key in model_type_dict.keys():
                file_name = array_path + '_' + key + "_ratio.npy"
                model_type_dict[key] = np.load(file_name)
                if key == 'chat':
                    plt.plot(layers,
                             model_type_dict[key],
                             label="Original")
                else:
                    plt.plot(layers,
                             model_type_dict[key],
                             label="Wanda" + key[11:])


            plt.legend(fontsize=16, loc='upper right',)
            plt.ylabel('Between class variance /'
                       '\nwithin-class variance', fontsize=18)
            plt.xlabel('Layer', fontsize=18)
            plt.tick_params(labelsize=18)
            plt.title(f'Separation between true and false'
                      f'\nstatements across layers',
                      fontsize=18)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        print("\n=>=> Finish running the step 1!\n")
    """
       Get the layer from which to extract activations
    based on the figure of the step 1.
    - Affirmative statements: as a sentence “stating that a 
    fact is so; answering ’yes’ to a question put or implied”. 
    - Negated statements: contain a negation like the word "not". 
    - Polarity of a statement as the grammatical category 
    indicating whether it is affirmative or negated.
    """
    layer = 12
    if run_step["step2"]:
        """
           Plot the Figure 3: Separation of true and false 
        statements along different truth directions as 
        measured by the AUROC. Figure 3 shows how well true 
        and false statements from different datasets separate 
        along t_G and t_P. For comparison, the authors trained 
        a Logistic Regression (LR) classifier with bias b = 0 
        on the centered activations a_ij^˜ = a_ij − µ_i. Its 
        direction d_LR separates true and false statements 
        similarly well as t_G.
        """
        print("\n=>=> You are running the step 2...")
        for model_type in model_type_list:
            print("\n=> For {}...".format(model_type))
            run_step2(train_sets=train_sets, train_set_sizes=train_set_sizes,
                      model_family=model_family, model_size=model_size,
                      model_type=model_type, layer=layer)


        print("\n=>=> Finish running the step 2!\n")


    if run_step["step3"]:
        """
           Activation vectors projected onto 2d truth subspace.
        Plot three figures:
        1: Figure 1:
        - Top left: The activation vectors of multiple statements
        projected onto the 2D subspace spanned by our estimates for t_G and t_P.
        - Top center: The activation vectors of affirmative true and false 
        statements separate along the direction t_A.
        - Top right: However, negated true and false statements do not 
        separate along t_A.
        - Bottom: Empirical distribution of activation vectors corresponding 
        to both affirmative and negated statements projected onto t_G and 
        t_A, respectively. Both affirmative and negated statements separate 
        well along the direction t_G proposed in this work.
        2: Figure 11: LLaMA2-13B: Left (a): Activations a_ij projected onto 
        t_G and t_P.
        3: Figure 7: The activation vectors of the larger_than and smaller_than 
        datasets projected onto t_G and t_P. In grey: the activation vectors of 
        statements from all affirmative and negated topic-specific datasets.
        """
        print("\n=>=> You are running the step 3...\n")
        for model_type in model_type_list:
            print("\n=> For {}...".format(model_type))
            t_g, t_p = run_step3(train_sets=train_sets,
                                 train_set_sizes=train_set_sizes,
                                 model_family=model_family,
                                 model_size=model_size,
                                 model_type=model_type, layer=layer)
            if run_step["step4"]:
                """
                   from probes import learn_truth_directions
                   t_g: General truth direction.
                   t_p: Polarity sensitive truth direction.
                   t_g, t_p = learn_truth_directions(
                       acts_centered=acts_centered,
                       labels=labels, polarities=polarities)
                   from train_sets = ["cities", "neg_cities", 
                      "sp_en_trans", "neg_sp_en_trans", 
                      "inventors", "neg_inventors", 
                      "animal_class", "neg_animal_class", 
                      "element_symb", "neg_element_symb", 
                      "facts", "neg_facts"]
                   Plot the Figure 4: The fraction of variance in the 
                centered and averaged activations ˜µ+i, ˜µ−i explained 
                by the Principal Components (PCs). Only the first 10 PCs 
                are shown.
                   To illustrate that When applying PCA to affirmative 
                statements only (top left), the first PC explains 
                approximately 60% of the variance in the centered and 
                averaged activations, with subsequent PCs contributing 
                significantly less, indicative of a one-dimensional 
                affirmative truth direction. Including both affirmative 
                and negated statements (top center) reveals a two-dimensional 
                truth subspace, where the first two PCs account 
                for more than 60% of the variance in the preprocessed 
                activations. Just two principal components sufficiently 
                capture the truth-related variance, suggesting only two 
                truth dimensions.
                   The authors also verified that these two PCs indeed 
                approximately correspond to t_G and t_P by computing the 
                cosine similarities between the first PC and t_G and 
                between the second PC and tP, measuring cosine 
                similarities of 0.98 and 0.97, respectively.
                """
                assert t_g is not None and t_p is not None
                print("\n=>=> You are running the step 4...\n")
                run_step4(model_family=model_family, model_size=model_size,
                          model_type=model_type, layer=layer, t_g=t_g, t_p=t_p)
                print("\n=>=> Finish running the step 4!\n")

        print("\n=>=> Finish running the step 3!\n")


    if run_step["step5"]:
        """
           Plot the Figure 5: Generalisation accuracies of truth directions 
        t before (left) and after (right) projecting out Span(t_G, t_P) from 
        the training activations. The x-axis is the training set and the 
        y-axis is the test set.
        """
        print("\n=>=> You are running the step 5...\n")
        for model_type in model_type_list:
            print("\n=> For {}...".format(model_type))
            run_step5(train_sets=train_sets, train_set_sizes=train_set_sizes,
                      model_family=model_family, model_size=model_size,
                      model_type=model_type, layer=layer)
        print("\n=>=> Finish running the step 5!\n")

    if run_step["step6"]:
        """
           Plot the left part of the Figure 5.
        """
        print("\n=>=> You are running the step 6...\n")
        for model_type in model_type_list:
            print("\n=> For {}...".format(model_type))
            run_step6(model_family=model_family, model_size=model_size,
                      model_type=model_type, layer=layer)

        print("\n=>=> Finish running the step 6!\n")



    return


if __name__ == '__main__':
    main()
