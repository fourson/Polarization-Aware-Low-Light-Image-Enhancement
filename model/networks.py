import functools

import torch
import torch.nn as nn
from torchvision import models


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ChannelShuffle(nn.Module):
    def __init__(self, groups=8):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.shape
        return x.reshape(N, self.groups, C // self.groups, H, W).transpose(1, 2).reshape(N, C, H, W)


class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, latent_dim=8, subsample=True):
        super(Self_Attn_FM, self).__init__()
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels=self.channel_latent, out_channels=in_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        if subsample:
            self.key_conv = nn.Sequential(
                self.key_conv,
                nn.MaxPool2d(2)
            )
            self.value_conv = nn.Sequential(
                self.value_conv,
                nn.MaxPool2d(2)
            )

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B x C x H x W)
            returns :
                out : self attention value + input feature
        """
        batchsize, C, height, width = x.size()
        c = self.channel_latent
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(x).view(batchsize, c, -1).permute(0, 2, 1)
        # proj_key: reshape to B x c x N_, N_ = H_ x W_
        proj_key = self.key_conv(x).view(batchsize, c, -1)
        # energy: B x N x N_, N = H x W, N_ = H_ x W_
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N_ x N, N = H x W, N_ = H_ x W_
        attention = self.softmax(energy).permute(0, 2, 1)
        # proj_value: B x c x N_, N_ = H_ x W_
        proj_value = self.value_conv(x).view(batchsize, c, -1)
        # attention_out: B x c x N, N = H x W
        attention_out = torch.bmm(proj_value, attention)
        # out: B x C x H x W
        out = self.out_conv(attention_out.view(batchsize, c, height, width))

        out = self.gamma * out + x
        return out


class Chuncked_Self_Attn_FM(nn.Module):
    """
        in_channel -> in_channel
    """

    def __init__(self, in_channel, latent_dim=8, subsample=True, grid=(8, 8)):
        super(Chuncked_Self_Attn_FM, self).__init__()

        self.self_attn_fm = Self_Attn_FM(in_channel, latent_dim=latent_dim, subsample=subsample)
        self.grid = grid

    def forward(self, x):
        N, C, H, W = x.shape
        chunk_size_H, chunk_size_W = H // self.grid[0], W // self.grid[1]
        x_ = x.reshape(N, C, self.grid[0], chunk_size_H, self.grid[1], chunk_size_W).permute(0, 2, 4, 1, 3, 5).reshape(
            N * self.grid[0] * self.grid[1], C, chunk_size_H, chunk_size_W)
        output = self.self_attn_fm(x_).reshape(N, self.grid[0], self.grid[1], C, chunk_size_H,
                                               chunk_size_W).permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return output


class DenseCell(nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size=3):
        super(DenseCell, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=growth_rate, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((x, self.conv_block(x)), dim=1)


class DenseBlock(nn.Module):
    """
        DenseBlock using bottleneck structure
        in_channel -> in_channel
    """

    def __init__(self, in_channel, growth_rate=32, n_blocks=3):
        super(DenseBlock, self).__init__()

        sequence = nn.ModuleList()

        dim = in_channel
        for i in range(n_blocks):
            sequence.append(DenseCell(dim, growth_rate))
            dim += growth_rate

        self.dense_cells = nn.Sequential(*sequence)
        self.fusion = nn.Conv2d(in_channels=dim, out_channels=in_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.fusion(self.dense_cells(x)) + x


class ResBlock(nn.Module):
    """
        ResBlock using bottleneck structure
        dim -> dim
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out


class AutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=6, norm_type='instance', use_dropout=False):
        super(AutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        ]

        dim = output_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.ReLU(inplace=True)
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            sequence += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(dim // 2),
                nn.ReLU(inplace=True)
            ]
            dim //= 2

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_layer, use_bias):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)


class SkipAutoencoderDownsamplingBlock(nn.Module):
    """
        Autoencoder downsampling block with skip links
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(SkipAutoencoderDownsamplingBlock, self).__init__()

        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]
        out_sequence += [nn.MaxPool2d(2)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        x_ = self.projection(x)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderUpsamplingBlock(nn.Module):
    """
        Autoencoder upsampling block with skip links
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(SkipAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        upsampled_x1 = self.upsample(x1)
        x_ = self.projection(torch.cat((x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone with skip links
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False,
                 use_channel_shuffle=True):
        super(SkipAutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_channel_shuffle)
            )
            dim *= 2

        dense_blocks_seq = n_blocks * [DenseBlock(dim)]
        self.dense_blocks = nn.Sequential(*dense_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                SkipAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias,
                                               use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out


class AttentionAutoencoderUpsamplingBlock(nn.Module):
    """
        Attention autoencoder upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, use_channel_shuffle):
        super(AttentionAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2, norm_layer, use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        x_ = self.projection(torch.cat((attentioned_x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class AttentionAutoencoderBackbone(nn.Module):
    """
        Attention autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False,
                 use_channel_shuffle=True):
        super(AttentionAutoencoderBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_channel_shuffle)
            )
            dim *= 2

        dense_blocks_seq = n_blocks * [DenseBlock(dim)]
        self.dense_blocks = nn.Sequential(*dense_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias,
                                                    use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out


class GlobalAvgPool(nn.Module):
    """(N,C,H,W) -> (N,C)"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, -1).mean(-1)


class SEBlock(nn.Module):
    """(N,C,H,W) -> (N,C,H,W)"""

    def __init__(self, in_channel, r):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_channel, in_channel // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // r, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return x * se_weight  # (N, C, H, W)
