import torchlib

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class OnlyImage(Dataset):

    def __init__(self, img_label_dataset):
        self.img_label_dataset = img_label_dataset

    def __len__(self):
        return len(self.img_label_dataset)

    def __getitem__(self, i):
        return self.img_label_dataset[i][0]


def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.MNIST('data/MNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]

    elif dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.FashionMNIST('data/FashionMNIST', transform=transform, download=True)
        img_shape = [32, 32, 1]

    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = datasets.CIFAR10('data/CIFAR10', transform=transform, download=True)
        img_shape = [32, 32, 3]

    else:
        raise NotImplementedError

    dataset = OnlyImage(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    return data_loader, img_shape


def make_celeba_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    crop_size = 108

    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchlib.DiskImageDataset(img_paths, map_fn=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


def make_anime_dataset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchlib.DiskImageDataset(img_paths, map_fn=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


# ==============================================================================
# =                               custom dataset                               =
# ==============================================================================

def make_custom_datset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, num_workers=4, pin_memory=False):
    transform = transforms.Compose([
        # ======================================
        # =               custom               =
        # ======================================
        ...,  # custom preprocessings
        # ======================================
        # =               custom               =
        # ======================================
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchlib.DiskImageDataset(img_paths, map_fn=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_remainder, pin_memory=pin_memory)

    img_shape = (resize, resize, 3)

    return data_loader, img_shape


# ==============================================================================
# =                                   debug                                    =
# ==============================================================================

# import imlib as im
# import numpy as np
# import pylib as py

# data_loader, _ = make_celeba_dataset(py.glob('data/img_align_celeba', '*.jpg'), batch_size=64)

# for img_batch in data_loader:
#     for img in img_batch.numpy():
#         img = np.transpose(img, (1, 2, 0))
#         im.imshow(img)
#         im.show()
