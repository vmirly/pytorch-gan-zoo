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

# models names
NAMESTR_VANILLA = 'vanilla'
NAMESTR_LSGAN = 'lsgan'
NAMESTR_WGAN = 'wgan'
NAMESTR_WGANCLIPPING = 'wgan-clipping'
NAMESTR_WGANGP = 'wgan-gp'
NAMESTR_WGANLP = 'wgan-lp'
NAMESTR_BEGAN = 'began'
NAMESTR_PIX2PIX = 'pix2pix'

# evaluation examples
NUM_EVAL_IMAGES = 64

# Network Parameters
Z_DIM = 100

FC_HIDDEN_DIMS = {
    NAMESTR_VANILLA: 128,
    NAMESTR_LSGAN: 128,
    NAMESTR_WGAN: 128,
    NAMESTR_WGANCLIPPING: 128,
    NAMESTR_WGANGP: 128,
    NAMESTR_WGANLP: 128,
    NAMESTR_BEGAN: None  # BEGAN has its own architecture
}

DROPOUT_PROBA = 0.5
PROBA_HFLIP = 0.5

NUM_CONV_FILTERS = {
    NAMESTR_VANILLA: 32,
    NAMESTR_LSGAN: 32,
    NAMESTR_WGAN: 32,
    NAMESTR_WGANCLIPPING: 32,
    NAMESTR_WGANGP: 32,
    NAMESTR_WGANLP: 32,
    NAMESTR_BEGAN: 32
}

DISC_ACTIVATIONS = {
    NAMESTR_VANILLA: 'sigmoid',
    NAMESTR_LSGAN: 'none',
    NAMESTR_WGAN: 'none',
    NAMESTR_WGANCLIPPING: 'none',
    NAMESTR_WGANGP: 'none',
    NAMESTR_WGANLP: 'none',
    NAMESTR_BEGAN: 'none'
}

# Optimizer
LEARNING_RATES = {
    NAMESTR_VANILLA: 0.001,
    NAMESTR_LSGAN: 0.001,
    NAMESTR_WGAN: 0.0001,
    NAMESTR_WGANCLIPPING: 0.0001,
    NAMESTR_WGANGP: 0.0001,
    NAMESTR_WGANLP: 0.0001,
    NAMESTR_BEGAN: 0.0001,
    NAMESTR_PIX2PIX: 0.0002
}
ADAM_BETA1 = 0.5
ADAM_BETA2 = {
    NAMESTR_VANILLA: 0.9,
    NAMESTR_LSGAN: 0.9,
    NAMESTR_WGAN: 0.9,
    NAMESTR_WGANCLIPPING: 0.9,
    NAMESTR_WGANGP: 0.9,
    NAMESTR_WGANLP: 0.9,
    NAMESTR_BEGAN: 0.9,
    NAMESTR_PIX2PIX: 0.999
}
