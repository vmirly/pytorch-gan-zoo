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

NUM_EVAL_IMAGES = 16

# Network Parameters
Z_DIM = 100

FC_HIDDEN_DIMS = {
    'vanilla': 128,
    'lsgan': 128,
    'wgan': 128,
    'wgan-clipping': 128,
    'wgan-gp': 128,
    'wgan-lp': 128,
    'began': None  # BEGAN has its own architecture
}

DROPOUT_PROBA = 0.5
PROBA_HFLIP = 0.5

NUM_CONV_FILTERS = {
    'vanilla': 32,
    'lsgan': 32,
    'wgan': 32,
    'wgan-clipping': 32,
    'wgan-gp': 32,
    'wgan-lp': 32,
    'began': 32
}

DISC_ACTIVATIONS = {
    'vanilla': 'sigmoid',
    'lsgan': 'none',
    'wgan': 'none',
    'wgan-clipping': 'none',
    'wgan-gp': 'none',
    'wgan-lp': 'none',
    'began': 'none'
}

# Optimizer
LEARNING_RATES = {
    'vanilla': 0.001,
    'lsgan': 0.001,
    'wgan': 0.0001,
    'wgan-clipping': 0.0001,
    'wgan-gp': 0.0001,
    'wgan-lp': 0.0001,
    'began': 0.0001,
    'pix2pix': 0.0002
}
ADAM_BETA1 = 0.5
ADAM_BETA2 = {
    'vanilla': 0.9,
    'lsgan': 0.9,
    'wgan': 0.9,
    'wgan-clipping': 0.9,
    'wgan-gp': 0.9,
    'wgan-lp': 0.9,
    'began': 0.9,
    'pix2pix': 0.999
}
