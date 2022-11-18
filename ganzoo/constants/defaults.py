"""
PyTorch GAN Zoo - Network Constants
Author: Vahid Mirjalili

List of keys for models:
  - 'vanilla': Vanilla GAN model
  - 'lsgan': Least-Square GAN
  - 'wgan': Wasserstein GAN (without weight clipping)
  - 'wgan-clipping': Wasserstein GAN with weight clipping
  - 'wgan-gp': Wasserstein GAN with gradient penalty
  - 'wgan-lp': Wasserstein GAN with Lipschitz penalty
  - 'began': Boundary Equilibrium GAN
"""
from ganzoo.constants import names

# FC networks
FC_HIDDEN_UNITS = 128

# evaluation examples
NUM_EVAL_IMAGES = 64

# Network Parameters
Z_DIM = 100

NUM_EPOCHS = 100
DROPOUT_PROBA = 0.5
PROBA_HFLIP = 0.5

FC_HIDDEN_DIMS = {  # TODO: remove if unused
    names.NAMESTR_VANILLA: 128,
    names.NAMESTR_LSGAN: 128,
    names.NAMESTR_WGAN: 128,
    names.NAMESTR_WGANCLIPPING: 128,
    names.NAMESTR_WGANGP: 128,
    names.NAMESTR_WGANLP: 128,
    names.NAMESTR_BEGAN: None  # BEGAN has its own architecture
}

NUM_CONV_FILTERS = {
    names.NAMESTR_VANILLA: 32,
    names.NAMESTR_LSGAN: 32,
    names.NAMESTR_WGAN: 32,
    names.NAMESTR_WGANCLIPPING: 32,
    names.NAMESTR_WGANGP: 32,
    names.NAMESTR_WGANLP: 32,
    names.NAMESTR_BEGAN: 32
}

DISC_ACTIVATIONS = {
    names.NAMESTR_VANILLA: 'sigmoid',
    names.NAMESTR_LSGAN: 'none',
    names.NAMESTR_WGAN: 'none',
    names.NAMESTR_WGANCLIPPING: 'none',
    names.NAMESTR_WGANGP: 'none',
    names.NAMESTR_WGANLP: 'none',
    names.NAMESTR_BEGAN: 'none'
}

# Optimizer
LEARNING_RATES = {
    names.NAMESTR_VANILLA: 0.001,
    names.NAMESTR_LSGAN: 0.001,
    names.NAMESTR_WGAN: 0.0001,
    names.NAMESTR_WGANCLIPPING: 0.0001,
    names.NAMESTR_WGANGP: 0.0001,
    names.NAMESTR_WGANLP: 0.0001,
    names.NAMESTR_BEGAN: 0.0001,
    names.NAMESTR_PIX2PIX: 0.0002
}
ADAM_BETA1 = 0.5
ADAM_BETA2 = {
    names.NAMESTR_VANILLA: 0.9,
    names.NAMESTR_LSGAN: 0.9,
    names.NAMESTR_WGAN: 0.9,
    names.NAMESTR_WGANCLIPPING: 0.9,
    names.NAMESTR_WGANGP: 0.9,
    names.NAMESTR_WGANLP: 0.9,
    names.NAMESTR_BEGAN: 0.9,
    names.NAMESTR_PIX2PIX: 0.999
}
