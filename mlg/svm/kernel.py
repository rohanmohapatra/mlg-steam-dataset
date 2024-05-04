import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    """Computes the RBF Kernel

    Parameters
    ----------
    """

    def __init__(self, gamma=1.0):
        super(RBFKernel, self).__init__()
        self.gamma = gamma

    def forward(self, x, y):
        # Compute the Euclidean distance between vectors
        distance = torch.cdist(x, y)
        # Compute the RBF kernel
        return torch.exp(-self.gamma * (distance**2))


class PolynomialKernel(nn.Module):
    """Computes the Polynomial Kernel

    Parameters
    ----------
    """

    def __init__(self, degree=3, gamma=1.0, coef0=1.0):
        super(PolynomialKernel, self).__init__()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def forward(self, x, y):
        # Polynomial kernel: (gamma * x^T * y + coef0) ** degree
        return (self.gamma * torch.matmul(x, y.T) + self.coef0) ** self.degree


class LinearKernel(nn.Module):
    """Computes the Linear Kernel

    Parameters
    ----------
    """

    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, x, y):
        # Linear kernel: x^T * y
        return torch.matmul(x, y.T)
