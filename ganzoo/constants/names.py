"""
"""

#================================#
#            Dataets             #
#================================#
# datasets names - strings
NAMESTR_MNIST = 'mnist'
NAMESTR_FASHIONMNIST = 'fashion-mnist'
NAMESTR_CIFAR10 = 'cifar10'
NAMESTR_CIFAR100 = 'cifar100'
NAMESTR_CELEBA = 'celebA'
NAMESTR_IMAGEFOLDER = 'custom-image-folder'

# list of downloadable datasets
DOWNLOADABLE_DATASETS = [
    NAMESTR_MNIST, NAMESTR_FASHIONMNIST,
    NAMESTR_CIFAR10, NAMESTR_CIFAR100,
    NAMESTR_CELEBA
]

#================================#
#         Distrbutions           #
#================================#
NAMESTR_UNIFORM = 'uniform'
NAMESTR_NORMAL = 'normal'
LIST_DISTRBUTIONS = [NAMESTR_UNIFORM, NAMESTR_NORMAL]

#================================#
#            MODELS              #
#================================#

# models names
NAMESTR_VANILLA = 'vanilla'
NAMESTR_LSGAN = 'lsgan'
NAMESTR_WGAN = 'wgan'
NAMESTR_WGANCLIPPING = 'wgan-clipping'
NAMESTR_WGANGP = 'wgan-gp'
NAMESTR_WGANLP = 'wgan-lp'
NAMESTR_BEGAN = 'began'
NAMESTR_PIX2PIX = 'pix2pix'

# FC networks
NAMESTR_FCSMALL = 'fc-small'
NAMESTR_FCSKIP = 'fc-skip'
NAMESTR_FCLARGE = 'fc-large'
LIST_FC_NETWORKS = [NAMESTR_FCSMALL, NAMESTR_FCSKIP, NAMESTR_FCLARGE]
