from torchvision import transforms
from torch.utils.data import DataLoader

from .dataset import dataset
from base.base_data_loader import BaseDataLoader


class TrainDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.TrainDataset(data_dir, transform=transform)

        super(TrainDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class InferDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.InferDataset(data_dir, transform=transform)

        super(InferDataLoader, self).__init__(self.dataset)


class GrayTrainDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.GrayTrainDataset(data_dir, transform=transform)

        super(GrayTrainDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class GrayInferDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
        ])
        self.dataset = dataset.GrayInferDataset(data_dir, transform=transform)

        super(GrayInferDataLoader, self).__init__(self.dataset)


class RealGrayInferDataLoader(DataLoader):
    def __init__(self, data_dir):
        self.dataset = dataset.RealGrayInferDataset(data_dir)

        super(RealGrayInferDataLoader, self).__init__(self.dataset)
