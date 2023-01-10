import os

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math

TonemapReinhard = cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork2':
        # 400ep
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


def tonemap(hdr_tensor):
    # tonemap hdr image tensor(N, C, H, W) for visualization
    tonemapped_tensor = torch.zeros(hdr_tensor.shape, dtype=torch.float32, requires_grad=False)
    for i in range(hdr_tensor.shape[0]):
        hdr = hdr_tensor[i].numpy().transpose((1, 2, 0))  # (H, W, C)
        is_rgb = (hdr.shape[2] == 3)
        if is_rgb:
            # if RGB (H, W, 3) , we should convert to an (H, W, 3) numpy array in order of BGR before tonemapping
            hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
        else:
            # if grayscale (H ,W, 1), we should copy the image 3 times to an (H, W, 3) numpy array before tonemapping
            hdr = cv2.merge([hdr, hdr, hdr])
        hdr = (hdr - np.min(hdr)) / (np.max(hdr) - np.min(hdr))
        tonemapped = TonemapReinhard.process(hdr)
        if is_rgb:
            # back to (C, H, W) tensor in order of RGB
            tonemapped_tensor[i] = torch.from_numpy(cv2.cvtColor(tonemapped, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        else:
            tonemapped_tensor[i] = torch.from_numpy(tonemapped[:, :, 0:1].transpose((2, 0, 1)))
    return tonemapped_tensor


def convert_to_colormap(img_tensor):
    # convert the grayscale image tensor(N, H, W) to colormap tensor(N, 3, H, W) for visualization
    N, H, W = img_tensor.shape
    colormap_tensor = torch.zeros((N, 3, H, W), dtype=torch.float32, requires_grad=False)
    for i in range(N):
        img = img_tensor[i].numpy()  # (H, W)
        img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)  # (H, W, 3) in BGR uint8
        img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255  # (3, H, W) in RGB float32
        colormap_tensor[i] = torch.from_numpy(img)
    return colormap_tensor


@torch.jit.script
def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor


@torch.jit.script
def convolve_with_kernel(img_tensor, kernel_tensor):
    # (N, C, H, W) image tensor and (h, w) kernel tensor -> (N, C, H, W) output tensor
    # kernel_tensor should be a buffer in the model to avoid runtime error when DataParallel
    # eg:
    # self.register_buffer('laplace_kernel',
    #                      torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, requires_grad=False),
    #                      persistent=False
    #                      )
    N, C, H, W = img_tensor.shape
    h, w = kernel_tensor.shape
    return F.conv2d(
        F.pad(img_tensor.reshape(N * C, 1, H, W),
              pad=[(h - 1) // 2, (h - 1) // 2, (w - 1) // 2, (w - 1) // 2],
              mode='reflect'
              ),
        kernel_tensor.reshape(1, 1, h, w)
    ).reshape(N, C, H, W)


@torch.jit.script
def compute_Si_from_Ii(I1, I2, I3, I4):
    S0 = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    DoP = torch.clamp(torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), min=0, max=1)  # in [0, 1]
    AoP = torch.atan2(S2, S1) / 2  # in [-pi/2, pi/2]
    AoP = (AoP < 0) * math.pi + AoP  # convert to [0, pi] by adding pi to negative values
    return S0, S1, S2, DoP, AoP


@torch.jit.script
def compute_Ii_from_Si(S0, S1, S2):
    I1 = (S0 - S1) / 2
    I2 = (S0 - S2) / 2
    I3 = (S0 + S1) / 2
    I4 = (S0 + S2) / 2
    DoP = torch.clamp(torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), min=0, max=1)  # in [0, 1]
    AoP = torch.atan2(S2, S1) / 2  # in [-pi/2, pi/2]
    AoP = (AoP < 0) * math.pi + AoP  # convert to [0, pi] by adding pi to negative values
    return I1, I2, I3, I4, DoP, AoP


@torch.jit.script
def compute_stokes(S0, DoP, AoP):
    S1 = S0 * DoP * torch.cos(2 * AoP)
    S2 = S0 * DoP * torch.sin(2 * AoP)
    return S0, S1, S2
