import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
