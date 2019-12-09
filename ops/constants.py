from losses import img_losses
from losses import gan_losses


GAN_LOSS = {
    'vanilla': (gan_losses.vanilla_gan_loss_G,
                gan_losses.vanialla_gan_loss_D_real,
                gan_losses.vanialla_gan_loss_D_real),
}

RECONSTRUCTION_LOSS = {
    'l1': img_losses.lossfn_l1,
    'l2': img_losses.lossfn_l2,
    'ce': None
}
