import torch
import torch.nn.functional as F

from utils.util import torch_laplacian
from .loss_utils.total_variation_loss import TVLoss

tv = TVLoss()


def l1_and_l2_and_tv(S0_out, S1_out, S2_out, S0, S1, S2, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = (F.l1_loss(S0_out, S0) + F.l1_loss(S1_out, S1) + F.l1_loss(S2_out, S2)) * l1_loss_lambda

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = (F.mse_loss(S0_out, S0) + F.mse_loss(S1_out, S1) + F.mse_loss(S2_out, S2)) * l2_loss_lambda

    tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    tv_loss = (tv(S0_out) + tv(S1_out) + tv(S2_out)) * tv_loss_lambda

    print('l1_loss:', l1_loss.item())
    print('l2_loss:', l2_loss.item())
    print('tv_loss:', tv_loss.item())
    return l1_loss + l2_loss + tv_loss


def l1_and_l2_and_tv_and_edge(S0_out, S1_out, S2_out, S0, S1, S2, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = (F.l1_loss(S0_out, S0) + F.l1_loss(S1_out, S1) + F.l1_loss(S2_out, S2)) * l1_loss_lambda

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = (F.mse_loss(S0_out, S0) + F.mse_loss(S1_out, S1) + F.mse_loss(S2_out, S2)) * l2_loss_lambda

    tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    tv_loss = (tv(S0_out) + tv(S1_out) + tv(S2_out)) * tv_loss_lambda

    edge_loss_lambda = kwargs.get('edge_loss_lambda', 1)
    edge_loss = (
                        F.mse_loss(torch.abs(torch_laplacian(S1_out)), torch.abs(torch_laplacian(S1))) +
                        F.mse_loss(torch.abs(torch_laplacian(S2_out)), torch.abs(torch_laplacian(S2)))
                ) * edge_loss_lambda

    print('l1_loss:', l1_loss.item())
    print('l2_loss:', l2_loss.item())
    print('tv_loss:', tv_loss.item())
    print('edge_loss:', edge_loss.item())
    return l1_loss + l2_loss + tv_loss + edge_loss
