from torch import nn

from .swish import Swish, H_Sigmoid, H_Swish


def make_act(act='ReLU', **kwargs):
    inplace = kwargs.pop("inplace", True)

    if len(act) == 0:
        return None
    act = {
        "ReLU": nn.ReLU(inplace=inplace),
        "ReLU6": nn.ReLU6(inplace=inplace),
        "PReLU": nn.PReLU(),
        "LeakyReLU": nn.LeakyReLU(inplace=inplace),
        "Sigmoid": nn.Sigmoid(),
        "H_Sigmoid": H_Sigmoid(),
        "Swish": Swish(),
        "H_Swish": H_Swish(),
    }[act]

    return act


class SeConv2d(nn.Module):
    def __init__(self, inplanes, innerplanse, inner_act='ReLU', out_act='Sigmoid'):
        super(SeConv2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, innerplanse, kernel_size=1),
            make_act(act=inner_act),
            nn.Conv2d(innerplanse, inplanes, kernel_size=1),
            make_act(act=out_act)
        )

        self._init_weights()

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y
