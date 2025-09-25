import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


def learn_truth_directions(acts_centered, labels, polarities):
    """
       In the section 3 of the paper, there is a procedure for
    supervised learning of t_G and t_P from the activations of
    affirmative and negated statements. Each activation
    vector a_ij is associated with a binary truth label
    τ_ij ∈ {−1,1} and a polarity pi ∈ {−1,1}.
    """


    """
       Check if all polarities are zero (handling both
    int and float) -> if yes learn only t_g.
    """
    all_polarities_zero = torch.allclose(polarities,
                                         torch.tensor([0.0]),
                                         atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0
    labels_copy = labels.clone()
    labels_copy = torch.where(labels_copy == 0.0,
                              torch.tensor(-1.0),
                              labels_copy)  # 0 to -1
    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = torch.column_stack([labels_copy,
                                labels_copy * polarities])

    """
       This optimization problem can be efficiently
    solved using ordinary least squares, yielding
    closed-form solutions for t_G and t_P for Eq.(5).
    """
    # torch.linalg.inv computes the inverse of the matrix
    solution = torch.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]  # General truth direction.
        t_p = solution[1, :]  # Polarity sensitive truth direction.

    return t_g, t_p


def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc


class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered=acts_centered,
                                              labels=labels,
                                              polarities=polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts=acts,
                                                        polarities=polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe
    
    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return torch.tensor(self.LR.predict(acts_2d))
    
    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d



def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = (
            torch.min(torch.stack((p_pos, p_neg), dim=-1),
                      dim=-1).values ** 2
    )
    return torch.mean(consistency_losses + confidence_losses)


class CCSProbe(torch.nn.Module):
    """
       Contrast Consistent Search (CCS) by Burns et al. [2023]:
    A method that identifies a direction satisfying logical consistency
    properties given contrast pairs of statements with opposite truth
    values. We create contrast pairs by pairing each affirmative
    statement with its negated counterpart, as done in
    Marks and Tegmark [2023].
    """
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=True),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)
    
    def pred(self, acts, iid=None):
        return self(acts).round()
    
    def from_data(acts, neg_acts, labels=None,
                  lr=0.001, weight_decay=0.1,
                  epochs=1000, device='cpu'):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)
        
        opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                                weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None:  # flip direction if needed
            labels = labels.to(device)
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]
    
    @property
    def bias(self):
        return self.net[0].bias.data[0]
    

class LRProbe():
    """
       Logistic Regression (LR): Used by Burns et al. [2023]
    and Marks and Tegmark [2023] to classify statements as true
    or false based on internal model activations and by
    Li et al. [2024] to find truthful directions.
    """
    def __init__(self):
        self.LR = None

    def from_data(acts, labels):
        probe = LRProbe()
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts.numpy(), labels.numpy())
        return probe

    def pred(self, acts):
        return torch.tensor(self.LR.predict(acts))
    

class MMProbe(torch.nn.Module):
    """
       Mass Mean (MM) probe by Marks and Tegmark [2023]:
    This method derives a truth direction tMM by calculating
    the difference between the mean of all true statements µ+ and
    the mean of all false statements µ−, such that tMM = µ+ − µ−.
    To ensure a fair comparison, we have extended the MM probe by
    incorporating a learned bias term. This bias is learned by
    fitting an LR classifier to the one-dimensional projections a^T t_MM.
    """
    def __init__(self, direction, LR):
        super().__init__()
        self.direction = direction
        self.LR = LR

    def forward(self, acts):
        proj = acts @ self.direction
        return torch.tensor(self.LR.predict(proj[:, None]))

    def pred(self, x):
        return self(x).round()

    def from_data(acts, labels, device='cpu'):
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None], labels)
        
        probe = MMProbe(direction, LR).to(device)

        return probe

