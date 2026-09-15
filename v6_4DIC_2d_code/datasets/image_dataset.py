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

        self.data_dir = os.path.join(self.root, 'train/')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_dic1_all = []
        self.vol_dic3_all = []
        self.vol_dic4_all = []
        self.vol_dic6_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_dic1 = f['dic1'][...]    # size, [416, 416]
                vol_dic3 = f['dic3'][...]
                vol_dic4 = f['dic4'][...]
                vol_dic6 = f['dic6'][...]
                vol_fluo  = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_dic1
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_dic1, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic1_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic4, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic4_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic6, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic6_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_dic1_all = np.concatenate(self.vol_dic1_all, 0)
        self.vol_dic3_all = np.concatenate(self.vol_dic3_all, 0)
        self.vol_dic4_all = np.concatenate(self.vol_dic4_all, 0)
        self.vol_dic6_all = np.concatenate(self.vol_dic6_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)  # [patch*16*25, 128, 128]


    def __getitem__(self, index):
        vol_dic1 = self.vol_dic1_all[index, ...]
        vol_dic3 = self.vol_dic3_all[index, ...]
        vol_dic4 = self.vol_dic4_all[index, ...]
        vol_dic6 = self.vol_dic6_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_dic1 = torch.from_numpy(vol_dic1.copy())
        vol_dic3 = torch.from_numpy(vol_dic3.copy())
        vol_dic4 = torch.from_numpy(vol_dic4.copy())
        vol_dic6 = torch.from_numpy(vol_dic6.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_dic1': vol_dic1,
                'vol_dic3': vol_dic3,
                'vol_dic4': vol_dic4,
                'vol_dic6': vol_dic6,
                'vol_fluo': vol_fluo,
                'opts_drop': True}

    def __len__(self):
        return self.vol_dic1_all.shape[0]



# ----------------------- Testing Dataset ---------------------
class Dataset_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_test
        self.n_patch = opts.n_patch_test
        self.test_pad = opts.test_pad

        self.data_dir = os.path.join(self.root, 'test/')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_dic1_all = []
        self.vol_dic3_all = []
        self.vol_dic4_all = []
        self.vol_dic6_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_dic1 = f['dic1'][...]    # size, [416, 416]
                vol_dic3 = f['dic3'][...]
                vol_dic4 = f['dic4'][...]
                vol_dic6 = f['dic6'][...]
                vol_fluo  = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_dic1
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_dic1, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic1_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic4, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic4_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic6, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic6_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_dic1_all = np.concatenate(self.vol_dic1_all, 0)
        self.vol_dic3_all = np.concatenate(self.vol_dic3_all, 0)
        self.vol_dic4_all = np.concatenate(self.vol_dic4_all, 0)
        self.vol_dic6_all = np.concatenate(self.vol_dic6_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)  # [patch*16*25, 128, 128]


    def __getitem__(self, index):
        vol_dic1 = self.vol_dic1_all[index, ...]
        vol_dic3 = self.vol_dic3_all[index, ...]
        vol_dic4 = self.vol_dic4_all[index, ...]
        vol_dic6 = self.vol_dic6_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_dic1 = torch.from_numpy(vol_dic1.copy())
        vol_dic3 = torch.from_numpy(vol_dic3.copy())
        vol_dic4 = torch.from_numpy(vol_dic4.copy())
        vol_dic6 = torch.from_numpy(vol_dic6.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_dic1': vol_dic1,
                'vol_dic3': vol_dic3,
                'vol_dic4': vol_dic4,
                'vol_dic6': vol_dic6,
                'vol_fluo': vol_fluo,
                'opts_drop': False}

    def __len__(self):
        return self.vol_dic1_all.shape[0]








# ----------------------- Validation Dataset ---------------------
class Dataset_Valid(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_valid
        self.n_patch = opts.n_patch_valid
        self.valid_pad = opts.valid_pad

        self.data_dir = os.path.join(self.root, 'valid/')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_dic1_all = []
        self.vol_dic3_all = []
        self.vol_dic4_all = []
        self.vol_dic6_all = []
        self.vol_fluo_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_dic1 = f['dic1'][...]    # size, [416, 416]
                vol_dic3 = f['dic3'][...]
                vol_dic4 = f['dic4'][...]
                vol_dic6 = f['dic6'][...]
                vol_fluo  = f['fluo'][...]

            # create the random index for cropping patches
            X_template = vol_dic1
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_dic1, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic1_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic3, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic3_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic4, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic4_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_dic6, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]  # [patch, 1, 128, 128]
            self.vol_dic6_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_fluo, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :]
            self.vol_fluo_all.append(X_patches)

        self.vol_dic1_all = np.concatenate(self.vol_dic1_all, 0)
        self.vol_dic3_all = np.concatenate(self.vol_dic3_all, 0)
        self.vol_dic4_all = np.concatenate(self.vol_dic4_all, 0)
        self.vol_dic6_all = np.concatenate(self.vol_dic6_all, 0)
        self.vol_fluo_all = np.concatenate(self.vol_fluo_all, 0)  # [patch*16*25, 128, 128]


    def __getitem__(self, index):
        vol_dic1 = self.vol_dic1_all[index, ...]
        vol_dic3 = self.vol_dic3_all[index, ...]
        vol_dic4 = self.vol_dic4_all[index, ...]
        vol_dic6 = self.vol_dic6_all[index, ...]
        vol_fluo = self.vol_fluo_all[index, ...]

        vol_dic1 = torch.from_numpy(vol_dic1.copy())
        vol_dic3 = torch.from_numpy(vol_dic3.copy())
        vol_dic4 = torch.from_numpy(vol_dic4.copy())
        vol_dic6 = torch.from_numpy(vol_dic6.copy())
        vol_fluo = torch.from_numpy(vol_fluo.copy())

        return {'vol_dic1': vol_dic1,
                'vol_dic3': vol_dic3,
                'vol_dic4': vol_dic4,
                'vol_dic6': vol_dic6,
                'vol_fluo': vol_fluo,
                'opts_drop': False}

    def __len__(self):
        return self.vol_dic1_all.shape[0]




if __name__ == '__main__':
    a = 1
