import torch
import torch.nn as nn


def weights_init(module):

    # classname = module.__class__.__name__

    if type(module) == nn.Linear:
        nn.init.uniform_(tensor=module.weight, a=0.0, b=1.0)
    elif type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
        nn.init.normal_(tensor=module.weight, mean=0.0, std=0.02)
    # if classname.find('Conv') != -1:
    #    nn.init.xavier_normal_(module.weight.data, gain=1.0)
    #    if module.bias is not None:
    #        nn.init.constant_(module.bias.data, 0)
    # elif classname.find('BatchNorm') != -1:
    #    nn.init.normal_(module.weight.data, 1.0, 0.02)
    #    nn.init.constant_(module.bias.data, 0)


class ConvBlock(nn.Module):
    """Convolutional Block with BN, ReLU."""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            name,
            stride=1,
            padding=0,
            bias=False,
            activation='relu',
            bn=False,
            transposed=False,
            dropout=False):

        super(ConvBlock, self).__init__()

        self.main = nn.Sequential()

        if not transposed:
            self.main.add_module(
                'Conv-{}'.format(name),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias))
        else:
            self.main.add_module(
                'tConv-{}'.format(name),
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=0,
                    bias=bias))

        if bn:
            self.main.add_module(
                'BatchNorm-'.format(name),
                nn.BatchNorm2d(
                    out_channels,
                    affine=True,
                    track_running_stats=True))

        if activation == 'relu':
            self.main.add_module(
                'relu-{}'.format(name),
                nn.ReLU(inplace=True))

        elif activation == 'leaky_relu':
            self.main.add_module(
                'lrelu-{}'.format(name),
                nn.LeakyReLU(inplace=True))

        self.main.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class Pix2PixGenerator(nn.Module):
    def __init__(
            self,
            n_inp_channels=3,
            n_out_channels=3,
            n_filters=64):

        super(Pix2PixGenerator, self).__init__()

        self.conv_first = ConvBlock(
            in_channels=n_inp_channels, out_channels=n_filters,
            kernel_size=1, name='first', bias=True, stride=1, padding=0,
            activation='relu', bn=False, dropout=False, transposed=False)

        # Contractive layers:
        self.conv1 = ConvBlock(
            in_channels=n_filters, out_channels=n_filters*2,
            kernel_size=3, name='c1', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=False)

        self.conv2 = ConvBlock(
            in_channels=n_filters*2, out_channels=n_filters*4,
            kernel_size=3, name='c2', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=False)

        self.conv3 = ConvBlock(
            in_channels=n_filters*4, out_channels=n_filters*8,
            kernel_size=3, name='c3', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=False)

        self.conv4 = ConvBlock(
            in_channels=n_filters*8, out_channels=n_filters*16,
            kernel_size=3, name='c4', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=False)

        self.conv5 = ConvBlock(
            in_channels=n_filters*16, out_channels=n_filters*32,
            kernel_size=3, name='c5', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=False)

        # Expansive layers:
        self.t_conv1 = ConvBlock(
            in_channels=n_filters*32, out_channels=n_filters*16,
            kernel_size=4, name='e1', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=True)

        self.t_conv2 = ConvBlock(
            in_channels=n_filters*32, out_channels=n_filters*8,
            kernel_size=4, name='e2', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=True)

        self.t_conv3 = ConvBlock(
            in_channels=n_filters*16, out_channels=n_filters*4,
            kernel_size=4, name='e3', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=True)

        self.t_conv4 = ConvBlock(
            in_channels=n_filters*8, out_channels=n_filters*2,
            kernel_size=4, name='e4', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=True)

        self.t_conv5 = ConvBlock(
            in_channels=n_filters*4, out_channels=n_filters,
            kernel_size=4, name='e4', bias=False, stride=2, padding=1,
            activation='relu', bn=True, dropout=False, transposed=True)

        # Last conv layer avec 1x1 kernel
        self.conv_last = ConvBlock(
            in_channels=n_filters, out_channels=n_out_channels,
            kernel_size=1, name='last', bias=True, stride=1, padding=0,
            activation=None, bn=False, dropout=False, transposed=False)

    def forward(self, x):
        # Conctractive layers:
        x0 = self.conv_first(x)
        x1 = self.conv1(x0)     # -> /2
        x2 = self.conv2(x1)     # -> /4
        x3 = self.conv3(x2)     # -> /8
        x4 = self.conv4(x3)     # -> /16
        x5 = self.conv5(x4)

        # Expansive layers:
        x6 = self.t_conv1(x5)   # -> /16
        x6c = torch.cat([x6, x4], axis=1)
        x7 = self.t_conv2(x6c)  # -> /4
        x7c = torch.cat([x7, x3], axis=1)
        x8 = self.t_conv3(x7c)  # -> /2
        x8c = torch.cat([x8, x2], axis=1)
        x9 = self.t_conv4(x8c)
        x9c = torch.cat([x9, x1], axis=1)
        x10 = self.t_conv5(x9c)

        out = self.conv_last(x10)

        return torch.tanh(out)


class Pix2PixDiscriminator(nn.Module):

    def __init__(
            self,
            n_inp_channels=6,
            n_filters=64):

        super(Pix2PixDiscriminator, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=n_inp_channels, out_channels=n_filters,
            kernel_size=3, name='d1', bias=True, stride=2, padding=1,
            activation='leaky_relu', bn=False, dropout=False, transposed=False)

        self.conv2 = ConvBlock(
            in_channels=n_filters, out_channels=n_filters*2,
            kernel_size=3, name='d2', bias=False, stride=2, padding=1,
            activation='leaky_relu', bn=True, dropout=False, transposed=False)

        self.conv3 = ConvBlock(
            in_channels=n_filters*2, out_channels=n_filters*4,
            kernel_size=3, name='d3', bias=False, stride=2, padding=1,
            activation='leaky_relu', bn=True, dropout=False, transposed=False)

        self.conv4 = ConvBlock(
            in_channels=n_filters*4, out_channels=n_filters*8,
            kernel_size=3, name='d4', bias=True, stride=2, padding=1,
            activation='leaky_relu', bn=False, dropout=False, transposed=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.sigmoid(x)

        return x


class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            kernel_size,
            padding,
            apply_bn=True,
            block_name=''):

        super(DownsampleBlock, self).__init__()

        self.main = nn.Sequential()
        self.main.add_module(
            name='conv_' + block_name,
            module=nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                stride=2, padding=padding))
        if apply_bn:
            self.main.add_module(
                name='bn_' + block_name,
                module=nn.BatchNorm2d(
                    num_features=n_filters,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True))
        self.main.add_module(
            name='lrelu_' + block_name,
            module=nn.LeakyReLU())

        self.main.apply(weights_init)

    def forward(self, x):
        return self.main(x)
