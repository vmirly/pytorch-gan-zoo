import torch.nn as nn


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
