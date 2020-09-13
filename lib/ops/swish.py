import torch
import torch.nn as nn
import torch.nn.functional as F


class H_Swish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class H_Sigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

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
