# this is for the arxiv_polytope publication
import torch
import numpy as np
import itertools
from torch.distributions import MultivariateNormal, Categorical

class MixtureOfGaussians(torch.distributions.Distribution):
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor, log_Z=0.0):
        """
        Initialize a mixture of Gaussians distribution.

        :param means: Tensor of shape (num_components, num_dimensions) representing the means of the components.
        :param covariances: Tensor of shape (num_components, num_dimensions, num_dimensions) representing the covariance matrices.
        :param weights: Tensor of shape (num_components,) representing the mixture weights. Should sum to 1.
        """
        assert means.device == covariances.device == weights.device, f'not same device'

        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.log_Z = log_Z

        # Number of components and dimensions
        self.num_components = means.shape[0]
        self.num_dimensions = means.shape[1]

        # Ensure weights sum to 1
        if not torch.isclose(weights.sum(), torch.tensor(1.0).to(device=weights.device, dtype=weights.dtype)):
            raise ValueError("Mixture weights must sum to 1.")

        # Define the categorical distribution for component selection
        self.categorical = Categorical(weights)

        # Define multivariate normal distributions for each component
        self.components = [
            MultivariateNormal(means[i], covariances[i])
            for i in range(self.num_components)
        ]

    def sample(self, sample_shape=torch.Size()):
        """
        Generate samples from the mixture distribution.

        :param sample_shape: Shape of the samples to generate.
        :return: Tensor of samples with shape (*sample_shape, num_dimensions).
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, )
        sample_shape = torch.Size(sample_shape)
        num_samples = sample_shape.numel() if sample_shape else 1

        # Sample component indices
        component_indices = self.categorical.sample(sample_shape)

        # Sample from the corresponding Gaussian components
        samples = torch.empty(num_samples, self.num_dimensions, device=self.means.device)
        for i in range(self.num_components):
            mask = component_indices == i
            num_samples_i = mask.sum().item()
            if num_samples_i > 0:
                samples[mask] = self.components[i].sample((num_samples_i,))

        return samples.squeeze(0) if sample_shape == torch.Size() else samples

    def log_prob(self, value):
        """
        Compute the log probability of a value under the mixture distribution.

        :param value: Tensor of shape (..., num_dimensions).
        :return: Tensor of log probabilities with shape (...,).
        """
        # Compute log probabilities for each component
        log_probs = torch.stack(
            [comp.log_prob(value) + torch.log(self.weights[i]) for i, comp in enumerate(self.components)],
            dim=-1
        )
        # Marginalize over components
        return torch.logsumexp(log_probs, dim=-1) - self.log_Z

    def copy_to(self, device):
        return MixtureOfGaussians(
            self.means.to(device=device), self.covariances.to(device=device), self.weights.to(device=device)
        )


def create_polytope(K, schlafli=None):
    """
    Create an H-representation (A, b) for a regular polytope in R^K
    specified by its Schläfli symbol.

    Parameters
    ----------
    K : int
        The ambient dimension.
    schlafli : tuple of ints
        The Schläfli symbol. For K=2, expect a one-tuple (p,)
        for a regular p-gon. For K>=3, expect a tuple of length K-1.
        We support:
          - Simplex: (3, 3, ..., 3)
          - Hypercube: (4, 3, ..., 3)
          - Cross-polytope: (3, 3, ..., 4)

    Returns
    -------
    A : ndarray, shape (m, K)
        Matrix such that each row defines a half-space: a_i^T x <= b_i.
    b : ndarray, shape (m,)
        Vector defining the right-hand side of the inequalities.

    Raises
    ------
    ValueError
        If the combination of K and schlafli symbol is not supported.
    """

    if schlafli is None:
        schlafli = np.ones(K - 1) * 3
        schlafli[0] = 4

    if K == 2 and len(schlafli) == 1:
        # Regular polygon in the plane.
        p = schlafli[0]
        A_list = []
        # For a regular polygon inscribed in the unit circle, the distance
        # from the center to each edge (the inradius) is cos(pi/p).
        inradius = np.cos(np.pi / p)
        for i in range(p):
            # Compute the outward normal for the i-th edge.
            # (Rotating by 2pi/p gives the correct periodicity.)
            theta = 2 * np.pi * i / p
            n = [np.cos(theta), np.sin(theta)]
            A_list.append(n)
        A = np.array(A_list)
        b = inradius * np.ones(p)
        return A, b

    elif len(schlafli) == K - 1:
        # For K>=3, we distinguish the three families.

        # 1. Simplex: Schläfli symbol (3,3,...,3)
        if all(s == 3 for s in schlafli):
            # We return the standard simplex:
            # { x in R^K : x_i >= 0, sum_{i=1}^K x_i <= 1 }
            A1 = -np.eye(K)  # x_i >= 0 becomes -x_i <= 0
            A2 = np.ones((1, K))  # sum_i x_i <= 1
            A = np.vstack([A1, A2])
            b = np.hstack([np.zeros(K), [1]])
            return A, b

        # 2. Hypercube: Schläfli symbol (4,3,...,3)
        elif schlafli[0] == 4 and all(s == 3 for s in schlafli[1:]):
            # The hypercube: { x in R^K : -1 <= x_i <= 1 }.
            # Its H-representation has 2K inequalities.
            A = np.vstack([np.eye(K), -np.eye(K)])
            b = np.ones(2 * K)
            return A, b

        # 3. Cross-polytope: Schläfli symbol (3,3,...,4)
        elif schlafli[-1] == 4 and all(s == 3 for s in schlafli[:-1]):
            # The cross-polytope (or orthoplex) in R^K is the convex hull of
            # the unit vectors and their negatives.
            # One H-representation is given by:
            # { x in R^K : sum_{i=1}^K s_i x_i <= 1 for all choices of s_i in {-1,1} }.
            # This yields 2^K inequalities.
            A_list = []
            for signs in itertools.product([-1, 1], repeat=K):
                A_list.append(list(signs))
            A = np.array(A_list)
            b = np.ones(2 ** K)
            return A, b

        else:
            raise ValueError("Unsupported Schläfli symbol for dimension {}.".format(K))
    else:
        raise ValueError("Dimension K and length of Schläfli symbol mismatch.")


if __name__ == "__main__":
    # Example usage
    device = 'cuda:0'
    means = torch.tensor([[0.0, 0.0], [5.0, 5.0]]).to(device)
    covariances = torch.stack([torch.eye(2), torch.eye(2) * 2]).to(device)
    weights = torch.tensor([0.4, 0.6]).to(device)

    mog = MixtureOfGaussians(means, covariances, weights)

    # Sample from the mixture
    samples = mog.sample(torch.Size((1000,)))

    # Compute log probability of a value
    log_prob = mog.log_prob(torch.tensor([1.0, 1.0]).to(device))
    print("Log Probability:", log_prob)
