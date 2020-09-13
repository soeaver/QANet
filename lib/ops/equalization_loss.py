import torch
from torch.nn import functional as F


def equalization_loss(logits, target, freq_info, lambda_=0.00177):
    """

    :param logits: predict class logits, exclude bg, [0, C-1]
    :param target: ground truth, including bg
    :param freq_info: frequence for each category
    :param lambda_: freq threshold
    :return: equalized loss
    """
    N, C = logits.size()
    bg_ind = C

    # expand_label
    expand_target = logits.new_zeros(N, C + 1)
    expand_target[torch.arange(N), target] = 1
    expand_target = expand_target[:, :C]

    # instance-level weight
    exclude_weight = (target != bg_ind).float()
    exclude_weight = exclude_weight.view(N, 1).expand(N, C)

    # class-level weight
    threshold_weight = logits.new_zeros(C)
    threshold_weight[freq_info < lambda_] = 1
    threshold_weight = threshold_weight.view(1, C).expand(N, C)

    eql_w = 1 - exclude_weight * threshold_weight * (1 - expand_target)

    loss = F.binary_cross_entropy_with_logits(logits, expand_target, reduction='none')

    return torch.sum(loss * eql_w) / N
