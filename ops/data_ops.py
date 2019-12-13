from PIL import Image
import numpy as np
import torch


def convert_tensor2image(
        t_image,
        unnormalize):

    if len(t_image.shape) == 0:
        t_image = torch.squeeze(t_image, dim=0)

    t_image = np.transpose(t_image, (1, 2, 0))
    if unnormalize:
        t_image = 0.5*t_image + torch.tensor([[[0.5, 0.5, 0.5]]])

    t_image = (t_image*255).numpy().astype('uint8')

    return Image.fromarray(t_image)
