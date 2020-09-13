import torch


class DICELoss:
    def __init__(self):
        super(DICELoss, self).__init__()
        pass

    def __call__(self, x, target):
        x = x.contiguous().view(x.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(x * target, 1)
        b = torch.sum(x * x, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d
