import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_keypoints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_keypoints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if target_weight is not None:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_keypoints
