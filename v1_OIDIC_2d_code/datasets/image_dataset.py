import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
from utils.function import *
import tifffile

# ----------------------- Training Dataset ---------------------
class Dataset_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_train
        self.n_patch = opts.n_patch_train

        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_oidic_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
            #  Unable to open file (unable to lock file, errno = 37, error message = 'No locks available')
                vol_oidic = f['oidic'][...]  # size, [416, 416]
                vol_fluo  = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_oidic
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_oidic, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_oidic_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_oidic_all = np.concatenate(self.vol_oidic_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)  # [patch*16*25, 128, 128]


    def __getitem__(self, index):
        vol_oidic = self.vol_oidic_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_oidic = torch.from_numpy(vol_oidic.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_oidic': vol_oidic,
                'vol_fluo': vol_fluo,
                'opts_drop': True}

    def __len__(self):
        return self.vol_oidic_all.shape[0]



# ----------------------- Testing Dataset ---------------------
class Dataset_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_test
        self.n_patch = opts.n_patch_test
        self.test_pad = opts.test_pad

        self.data_dir = os.path.join(self.root, 'test')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_oidic_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_oidic = f['oidic'][...]
                vol_fluo = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_oidic  # size: [64, 64, 64]
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_oidic, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # (n_patch=1,1,64,64,64)
            self.vol_oidic_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_oidic_all = np.concatenate(self.vol_oidic_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)

    def __getitem__(self, index):
        vol_oidic = self.vol_oidic_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_oidic = torch.from_numpy(vol_oidic.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_oidic': vol_oidic,
                'vol_fluo': vol_fluo,
                'opts_drop': False}  # [1,64,64,64]

    def __len__(self):
        return self.vol_oidic_all.shape[0]








# ----------------------- Validation Dataset ---------------------
class Dataset_Valid(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_valid
        self.n_patch = opts.n_patch_valid
        self.valid_pad = opts.valid_pad

        self.data_dir = os.path.join(self.root, 'valid')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_oidic_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_oidic = f['oidic'][...]
                vol_fluo = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_oidic  # size: [424, 424]
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_oidic, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # (n_patch=1,1,64,64,64)
            self.vol_oidic_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_oidic_all = np.concatenate(self.vol_oidic_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)

    def __getitem__(self, index):
        vol_oidic = self.vol_oidic_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_oidic = torch.from_numpy(vol_oidic.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_oidic': vol_oidic,
                'vol_fluo': vol_fluo,
                'opts_drop': False}  # [1,64,64,64]

    def __len__(self):
        return self.vol_oidic_all.shape[0]




if __name__ == '__main__':
    a = 1
