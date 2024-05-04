import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, targets):
        # Hinge loss: max(0, 1 - y * f(x))
        return torch.mean(torch.clamp(1 - outputs * targets, min=0))
