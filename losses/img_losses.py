import torch.nn.functional as F


def l1_lossfn(input, target):
    return F.l1_loss(input, target)


def l2_lossfn(input, target):
    return F.mse_loss(input, target)
