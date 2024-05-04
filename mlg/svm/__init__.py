from typing import Literal

import torch
import torch.nn as nn

from mlg.svm.kernel import LinearKernel, PolynomialKernel, RBFKernel


class BaseSVM(nn.Module):
    def __init__(self, num_samples) -> None:
        self.kernel = None
        self.weight = nn.Parameter(torch.zeros(num_samples))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, support_vectors):
        if self.kernel is None:
            raise ValueError("Kernel cannot be empty")

        # Calculate the kernel values between support vectors and input samples
        kernel_vals = self.kernel(x, support_vectors)
        # Prediction is a linear combination of support vector weights and kernel values
        outputs = torch.matmul(kernel_vals, self.weight) + self.bias


class RBFSVM(BaseSVM):
    def __init__(self, num_samples, gamma=1.0):
        super(RBFSVM, self).__init__(num_samples)
        self.kernel = RBFKernel(gamma)


class PolynomialSVM(BaseSVM):
    def __init__(self, num_samples, degree=3, gamma=1.0, coef0=1.0):
        super(PolynomialSVM, self).__init__(num_samples)
        self.kernel = PolynomialKernel(degree, gamma, coef0)


class LinearSVM(BaseSVM):
    def __init__(self, num_samples):
        super(LinearSVM, self).__init__(num_samples)
        self.kernel = LinearKernel()


class SVM(BaseSVM):
    def __init__(
        self,
        num_samples,
        kernel: Literal["linear", "rbf", "poly"],
        degree=3,
        gamma=1.0,
        coef0=1.0,
    ):
        super().__init__(num_samples)
        if kernel == "linear":
            self.kernel = LinearKernel()
        elif kernel == "poly":
            self.kernel = PolynomialKernel(degree, gamma, coef0)
        elif kernel == "rbf":
            self.kernel = RBFKernel(gamma)
        else:
            raise ValueError("Not implemented yet")
