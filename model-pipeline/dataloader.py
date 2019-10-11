import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data


class HDF5Dataset_1(data.Dataset):

    def __init__(self, file_path, transform=None):
        '''Loop over all hdf5 files and initialize list of hdf5 datasets.
        HDF5 datasets are not stored in the RAM.'''

        self.archive = list()

        p = Path(file_path)
        files = sorted(p.glob('*.hdf5'))
        print(files)

        self.length = 0

        for idx, file in enumerate(files):
            hf5 = h5py.File(file, 'r')
            self.archive.append(hf5)
            self.length += self.archive[idx]['raw_image'].shape[0]

        if os.path.exists('./data/data.hdf5'):
            os.remove('./data/data.hdf5')

        self.datafile = h5py.File('./data/data.hdf5', mode='w')
        self.datafile.create_dataset("raw_images", (self.length, 20, 7, 25, 116), np.float32)
        self.datafile.create_dataset("heatmaps", (self.length, 130, 100, 13), np.float32)
        current_images = 0
        for idx, file in enumerate(files):
            raw_images = self.archive[idx]['raw_image'][()]
            heatmaps = self.archive[idx]['heatmap'][()]

            num_new_images = raw_images.shape[0]

            self.datafile['raw_images'][current_images: current_images + num_new_images] = raw_images
            self.datafile['heatmaps'][current_images: current_images + num_new_images] = heatmaps
            current_images += num_new_images

        # if transform:
        #     self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[d1_mean, d2_mean],
        #                          std=[d1_std, d2_std])
        #     ])

    def __getitem__(self, index):
        ''''''
        x = torch.from_numpy(self.datafile['raw_images'][index])
        y = torch.from_numpy(self.datafile['heatmaps'][index])
        # if self.transform:
        #     y = self.transform()
        y = y.permute(2, 1, 0)

        return x, y

    def __len__(self):
        return self.length


class HDF5Dataset_3(data.Dataset):

    def __init__(self, file_path, transform=None):
        '''Loop over all hdf5 files and initialize list of hdf5 datasets.
        HDF5 datasets are not stored in the RAM.'''

        self.archive = list()

        p = Path(file_path)
        files = sorted(p.glob('*.hdf5'))

        print(files)

        self.batch_count = 0

        self.length = 0

        for idx, file in enumerate(files):
            hf5 = h5py.File(file, 'r')
            self.archive.append(hf5)
            self.length += self.archive[idx]['raw_image'].shape[0]

        if os.path.exists('./data/data.hdf5'):
            os.remove('./data/data.hdf5')

        self.datafile = h5py.File('./data/data.hdf5', mode='w')
        self.datafile.create_dataset("raw_images", (self.length, 20, 7, 25, 116), np.float32)
        self.datafile.create_dataset("heatmaps", (self.length, 43, 58, 13), np.float32)
        current_images = 0
        for idx, file in enumerate(files):
            raw_images = self.archive[idx]['raw_image'][()]
            heatmaps = self.archive[idx]['heatmap'][()]

            num_new_images = raw_images.shape[0]

            self.datafile['raw_images'][current_images: current_images + num_new_images] = raw_images
            self.datafile['heatmaps'][current_images: current_images + num_new_images] = heatmaps
            current_images += num_new_images

    def __getitem__(self, index):
        ''''''

        x = (self.datafile['raw_images'][index])

        x1 = x[:, :, 0:24, :]
        x_1 = np.sum(x1, axis=1)

        x_2 = np.sum(x, axis=2)
        x_2 = np.array([np.append(x, np.zeros((1, 116)), axis=0) for x in x_2]).astype(np.float)

        x_2 = torch.from_numpy(x_2)
        x_1 = torch.from_numpy(x_1)

        y = torch.from_numpy(self.datafile['heatmaps'][index])
        y = y.permute(2, 0, 1)

        return x_1, x_2, y

    def __len__(self):
        return self.length


class HDF5Dataset_RPN(data.Dataset):

    def __init__(self, file_path, transform=None):
        '''Loop over all hdf5 files and initialize list of hdf5 datasets.
        HDF5 datasets are not stored in the RAM.
        As datasets are closed after being loaded, a temporary dataset is created called 'data.hdf5'
        This is used during training to load the data. Only one batch is then loaded into the RAM.


        Proposal Network: Heatmap contains all keypoints. Keypooints are given as extra dataset'''

        self.archive = list()

        p = Path(file_path)
        files = sorted(p.glob('*.hdf5'))

        print(files)

        self.length = 0

        for idx, file in enumerate(files):
            hf5 = h5py.File(file, 'r')
            self.archive.append(hf5)
            self.length += self.archive[idx]['raw_image'].shape[0]

        if os.path.exists('./data/data.hdf5'):
            os.remove('./data/data.hdf5')

        self.datafile = h5py.File('./data/data.hdf5', mode='w')
        self.datafile.create_dataset("raw_images", (self.length, 20, 7, 25, 116), np.float32)
        self.datafile.create_dataset("heatmap", (self.length, 43, 58, 13), np.float32)
        current_images = 0
        for idx, file in enumerate(files):
            raw_images = self.archive[idx]['raw_image'][()]
            heatmap = self.archive[idx]['heatmap'][()]
            keypoints = self.archive[idx]['heatmap'][()]

            num_new_images = raw_images.shape[0]

            self.datafile['raw_images'][current_images: current_images + num_new_images] = raw_images
            self.datafile['heatmap'][current_images: current_images + num_new_images] = heatmap
            current_images += num_new_images

    def __getitem__(self, index):
        ''''''
        x = torch.from_numpy(self.datafile['raw_images'][index])
        y = torch.from_numpy(self.datafile['heatmap'][index])
        y = y.permute(2, 0, 1)

        return x, y

    def __len__(self):
        return self.length


class HDF5Dataset_2(data.Dataset):

    def __init__(self, archive):
        self.archive = h5py.File(archive, 'r')
        self.raw_images = self.archive['raw_image']
        self.heatmaps = self.archive['heatmap']

        self.input_shape = self.raw_images[0].shape
        self.output_shape = self.heatmaps[0].shape

        print('Input shape:')
        print(self.input_shape)

        self.length = len(self.raw_images)

    def __getitem__(self, index):
        x = torch.from_numpy(self.raw_images[index]).float()
        y = torch.from_numpy(self.heatmaps[index]).float()
        # y = y.permute(2, 1, 0)

        return x, y

    def __len__(self):
        return self.length
