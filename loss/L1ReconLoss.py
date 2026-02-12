import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.registry import LOSSES

############################################################
#  Basic Recon Losses
############################################################

@LOSSES.register
class L1ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSSES.register
class MSEReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSSES.register
class SmoothL1Loss(nn.Module):
    """Huber loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target)
