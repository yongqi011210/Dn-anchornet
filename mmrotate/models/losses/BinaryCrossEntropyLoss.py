import torch.nn as nn
from ..builder import ROTATED_LOSSES

@ROTATED_LOSSES.register_module()
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss * self.loss_weight
