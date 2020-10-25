import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishX(nn.Module):
    def __init__(self, maxvalue=2.72):
        super(SwishX, self).__init__()
        self.maximal = nn.Parameter(torch.FloatTensor([maxvalue]))

    def forward(self, x):
        output = x * torch.sigmoid(x)
        output = output.sub(self.maximal).clamp(max=0.0).add(self.maximal)
        return output
