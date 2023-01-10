import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from utils.util import torch_laplacian
from .networks import Chuncked_Self_Attn_FM, DenseBlock, AttentionAutoencoderBackbone, get_norm_layer


class DefaultModel(BaseModel):
    """
        for learning S0, S1, and S2 (reconstruct Stokes parameters)
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False):
        super(DefaultModel, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction_S0 = nn.Sequential(
            nn.Conv2d(3, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.feature_extraction_S12_edge = nn.Sequential(
            nn.Conv2d(3, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.ReLU(True),
            Chuncked_Self_Attn_FM(in_channel=init_dim // 2, latent_dim=8, subsample=True, grid=(8, 8)),
            DenseBlock(in_channel=init_dim // 2, growth_rate=32, n_blocks=3)
        )
        self.feature_extraction_S12 = nn.Sequential(
            nn.Conv2d(3, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.ReLU(True)
        )
        self.backbone_S0 = AttentionAutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=2, n_blocks=3,
                                                        norm_type=norm_type, use_dropout=use_dropout,
                                                        use_channel_shuffle=True)
        self.backbone_S12 = AttentionAutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=2, n_blocks=3,
                                                         norm_type=norm_type, use_dropout=use_dropout,
                                                         use_channel_shuffle=True)
        self.out_block_S0 = nn.Sequential(
            nn.Conv2d(init_dim, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.out_block_S12 = nn.Sequential(
            nn.Conv2d(init_dim, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, S0_in, S1_in, S2_in):
        # both S0_in, S1_in, and S2_in have already been normalized to [0, 1]

        # for S0
        feature_S0 = self.feature_extraction_S0(S0_in)
        backbone_out_S0 = self.backbone_S0(feature_S0)
        S0_out = self.out_block_S0(backbone_out_S0) + S0_in
        S0_out = torch.clamp(S0_out, min=0, max=1)

        # for S1
        feature_S1_edge = self.feature_extraction_S12_edge(torch.abs(torch_laplacian(S1_in)))
        feature_S1 = self.feature_extraction_S12(S1_in)
        backbone_out_S1 = self.backbone_S12(torch.cat((feature_S1_edge, feature_S1), dim=1))
        S1_out = self.out_block_S12(backbone_out_S1) + S1_in
        S1_out = torch.clamp(S1_out, min=0, max=1)

        # for S2
        feature_S2_edge = self.feature_extraction_S12_edge(torch.abs(torch_laplacian(S2_in)))
        feature_S2 = self.feature_extraction_S12(S2_in)
        backbone_out_S2 = self.backbone_S12(torch.cat((feature_S2_edge, feature_S2), dim=1))
        S2_out = self.out_block_S12(backbone_out_S2) + S2_in
        S2_out = torch.clamp(S2_out, min=0, max=1)

        return S0_out, S1_out, S2_out


class GrayDefaultModel(BaseModel):
    """
        for learning S0, S1, and S2 (reconstruct Stokes parameters)
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False):
        super(GrayDefaultModel, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction_S0 = nn.Sequential(
            nn.Conv2d(1, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.feature_extraction_S12_edge = nn.Sequential(
            nn.Conv2d(1, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.ReLU(True),
            Chuncked_Self_Attn_FM(in_channel=init_dim // 2, latent_dim=8, subsample=True, grid=(8, 8)),
            DenseBlock(in_channel=init_dim // 2, growth_rate=32, n_blocks=3)
        )
        self.feature_extraction_S12 = nn.Sequential(
            nn.Conv2d(1, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_dim // 2),
            nn.ReLU(True)
        )
        self.backbone_S0 = AttentionAutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=2, n_blocks=3,
                                                        norm_type=norm_type, use_dropout=use_dropout,
                                                        use_channel_shuffle=True)
        self.backbone_S12 = AttentionAutoencoderBackbone(init_dim, output_nc=init_dim, n_downsampling=2, n_blocks=3,
                                                         norm_type=norm_type, use_dropout=use_dropout,
                                                         use_channel_shuffle=True)
        self.out_block_S0 = nn.Sequential(
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.out_block_S12 = nn.Sequential(
            nn.Conv2d(init_dim, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, S0_in, S1_in, S2_in):
        # both S0_in, S1_in, and S2_in have already been normalized to [0, 1]

        # for S0
        feature_S0 = self.feature_extraction_S0(S0_in)
        backbone_out_S0 = self.backbone_S0(feature_S0)
        S0_out = self.out_block_S0(backbone_out_S0) + S0_in
        S0_out = torch.clamp(S0_out, min=0, max=1)

        # for S1
        feature_S1_edge = self.feature_extraction_S12_edge(torch.abs(torch_laplacian(S1_in)))
        feature_S1 = self.feature_extraction_S12(S1_in)
        backbone_out_S1 = self.backbone_S12(torch.cat((feature_S1_edge, feature_S1), dim=1))
        S1_out = self.out_block_S12(backbone_out_S1) + S1_in
        S1_out = torch.clamp(S1_out, min=0, max=1)

        # for S2
        feature_S2_edge = self.feature_extraction_S12_edge(torch.abs(torch_laplacian(S2_in)))
        feature_S2 = self.feature_extraction_S12(S2_in)
        backbone_out_S2 = self.backbone_S12(torch.cat((feature_S2_edge, feature_S2), dim=1))
        S2_out = self.out_block_S12(backbone_out_S2) + S2_in
        S2_out = torch.clamp(S2_out, min=0, max=1)

        return S0_out, S1_out, S2_out