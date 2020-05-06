import torch
import torch.nn as nn


def onehot_embedding(num_classes, device=None):
    y = torch.eye(num_classes)
    if device is not None:
        y = y.to(device)

    def encode_to_onehot(labels):
        return y[labels]

    return encode_to_onehot


class Flatten(nn.Module):
    """
    A custom layer to flatten the feature maps
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    """
    A custom layer for reshaping feature maps
    """
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, input):
        return input.view(input.size(0), *self.new_shape)
