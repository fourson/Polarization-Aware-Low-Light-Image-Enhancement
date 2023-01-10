import numpy as np
import os
import fnmatch

import cv2
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        as input:
        L1, L2, L3, L4: four low light polarized images, [0, 1], as float32
        amp: amplification, [0, inf], as float

        as target:
        I1, I2, I3, I4: four enhanced polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')  # {name}_{time_short}.png
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.I1_dir = os.path.join(data_dir, 'I1')  # {name}_{time_long}.png
        self.I2_dir = os.path.join(data_dir, 'I2')
        self.I3_dir = os.path.join(data_dir, 'I3')
        self.I4_dir = os.path.join(data_dir, 'I4')

        self.Li_file_names = sorted(fnmatch.filter(os.listdir(self.L1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])
        self.Ii_file_names = sorted(fnmatch.filter(os.listdir(self.I1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])

        self.transform = transform

    def __len__(self):
        return len(self.Li_file_names)

    def __getitem__(self, index):
        # as input:
        Li_file_name = self.Li_file_names[index]
        name, time_short = Li_file_name[:-4].rsplit('_', 1)
        time_short = float(time_short)
        # (H, W, 3)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, Li_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, Li_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, Li_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, Li_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # as target:
        Ii_file_name = self.Ii_file_names[index]
        time_long = Ii_file_name[:-4].rsplit('_', 1)[1]
        time_long = float(time_long)
        # (H, W, 3)
        I1 = cv2.cvtColor(cv2.imread(os.path.join(self.I1_dir, Ii_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I2 = cv2.cvtColor(cv2.imread(os.path.join(self.I2_dir, Ii_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I3 = cv2.cvtColor(cv2.imread(os.path.join(self.I3_dir, Ii_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I4 = cv2.cvtColor(cv2.imread(os.path.join(self.I4_dir, Ii_file_name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # (1, 1, 1)
        amp = time_long / time_short * torch.ones((1, 1, 1))

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

            I1 = self.transform(I1)
            I2 = self.transform(I2)
            I3 = self.transform(I3)
            I4 = self.transform(I4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'I1': I1, 'I2': I2, 'I3': I3, 'I4': I4, 'amp': amp,
                'name': name}


class InferDataset(Dataset):
    """
        as input:
        L1, L2, L3, L4: four low light polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')  # {name}_{amp}x.png
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.Li_file_names = sorted(fnmatch.filter(os.listdir(self.L1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])

        self.transform = transform

    def __len__(self):
        return len(self.Li_file_names)

    def __getitem__(self, index):
        # as input:
        Li_file_name = self.Li_file_names[index]
        name, amp_str = Li_file_name[:-4].rsplit('_', 1)
        # (H, W, 3)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, Li_file_name)),
                          cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, Li_file_name)),
                          cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, Li_file_name)),
                          cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, Li_file_name)),
                          cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # (1, 1, 1)
        amp = float(amp_str[:-1]) * torch.ones((1, 1, 1))

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'amp': amp, 'name': name}


class GrayTrainDataset(Dataset):
    """
        for grayscale

        as input:
        L1, L2, L3, L4: four low light polarized images, [0, 1], as float32
        amp: amplification, [0, inf], as float

        as target:
        I1, I2, I3, I4: four enhanced polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')  # {name}_{time_short}.png
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.I1_dir = os.path.join(data_dir, 'I1')  # {name}_{time_long}.png
        self.I2_dir = os.path.join(data_dir, 'I2')
        self.I3_dir = os.path.join(data_dir, 'I3')
        self.I4_dir = os.path.join(data_dir, 'I4')

        self.Li_file_names = sorted(fnmatch.filter(os.listdir(self.L1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])
        self.Ii_file_names = sorted(fnmatch.filter(os.listdir(self.I1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])

        self.transform = transform

    def __len__(self):
        return len(self.Li_file_names)

    def __getitem__(self, index):
        # as input:
        Li_file_name = self.Li_file_names[index]
        name, time_short = Li_file_name[:-4].rsplit('_', 1)
        time_short = float(time_short)
        # (H, W, 1)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8

        # as target:
        Ii_file_name = self.Ii_file_names[index]
        time_long = Ii_file_name[:-4].rsplit('_', 1)[1]
        time_long = float(time_long)
        # (H, W, 1)
        I1 = cv2.cvtColor(cv2.imread(os.path.join(self.I1_dir, Ii_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        I2 = cv2.cvtColor(cv2.imread(os.path.join(self.I2_dir, Ii_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        I3 = cv2.cvtColor(cv2.imread(os.path.join(self.I3_dir, Ii_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        I4 = cv2.cvtColor(cv2.imread(os.path.join(self.I4_dir, Ii_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8

        # (1, 1, 1)
        amp = time_long / time_short * torch.ones((1, 1, 1))

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

            I1 = self.transform(I1)
            I2 = self.transform(I2)
            I3 = self.transform(I3)
            I4 = self.transform(I4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'I1': I1, 'I2': I2, 'I3': I3, 'I4': I4, 'amp': amp,
                'name': name}


class GrayInferDataset(Dataset):
    """
        for grayscale

        as input:
        L1, L2, L3, L4: four low light polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')  # {name}_{amp}x.png
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.Li_file_names = sorted(fnmatch.filter(os.listdir(self.L1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])

        self.transform = transform

    def __len__(self):
        return len(self.Li_file_names)

    def __getitem__(self, index):
        # as input:
        Li_file_name = self.Li_file_names[index]
        name, amp_str = Li_file_name[:-4].rsplit('_', 1)
        # (H, W, 3)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, Li_file_name)), cv2.COLOR_BGR2GRAY)[:, :,
             None]  # [0, 255] uint8

        # (1, 1, 1)
        amp = float(amp_str[:-1]) * torch.ones((1, 1, 1))

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'amp': amp, 'name': name}


class RealGrayInferDataset(Dataset):
    """
        for grayscale (captured directly from PolCamGray)

        as input:
        L1, L2, L3, L4: four low light polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir):
        self.L1_dir = os.path.join(data_dir, 'L1')  # {name}_{amp}x.png
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.Li_file_names = sorted(fnmatch.filter(os.listdir(self.L1_dir), '*.png'), key=lambda x: x.rsplit('_', 1)[0])

    def __len__(self):
        return len(self.Li_file_names)

    def __getitem__(self, index):
        # as input:
        Li_file_name = self.Li_file_names[index]
        name, amp_str = Li_file_name[:-4].rsplit('_', 1)
        # (H, W, 3)
        L1 = cv2.imread(os.path.join(self.L1_dir, Li_file_name), -1)[:, :, None].transpose(2, 0, 1)  # [0, 65535] uint16
        L2 = cv2.imread(os.path.join(self.L2_dir, Li_file_name), -1)[:, :, None].transpose(2, 0, 1)  # [0, 65535] uint16
        L3 = cv2.imread(os.path.join(self.L3_dir, Li_file_name), -1)[:, :, None].transpose(2, 0, 1)  # [0, 65535] uint16
        L4 = cv2.imread(os.path.join(self.L4_dir, Li_file_name), -1)[:, :, None].transpose(2, 0, 1)  # [0, 65535] uint16

        L1 = torch.from_numpy(np.float32(L1) / 65535)  # [0, 1] float32 tensor
        L2 = torch.from_numpy(np.float32(L2) / 65535)  # [0, 1] float32 tensor
        L3 = torch.from_numpy(np.float32(L3) / 65535)  # [0, 1] float32 tensor
        L4 = torch.from_numpy(np.float32(L4) / 65535)  # [0, 1] float32 tensor

        # (1, 1, 1)
        amp = float(amp_str[:-1]) * torch.ones((1, 1, 1))

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'amp': amp, 'name': name}
