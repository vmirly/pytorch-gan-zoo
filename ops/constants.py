from losses import img_losses
from losses import gan_losses


GAN_LOSS = {
    'vanilla': (gan_losses.vanilla_gan_lossfn_G,
                gan_losses.vanilla_gan_lossfn_D_real,
                gan_losses.vanilla_gan_lossfn_D_fake),
}

RECONSTRUCTION_LOSS = {
    'l1': img_losses.l1_lossfn,
    'l2': img_losses.l2_lossfn,
    'ce': None
}
