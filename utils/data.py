"""
    Custom PyTorch data classes (Datasets, DataLoaders) used in d-SNE
    training and testing procedures.
"""

# Stdlib imports
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Normalize


def set_paths(root, dataset_name, phase):
    keys = ['{}'.format(int(i)) for i in range(10)]
    values = [[] for _ in range(10)]
    path_dict = dict(zip(keys, values))
    for i in range(10):
        imgs_path = sorted(glob.glob(os.path.join(root, dataset_name, phase, '{}'.format(int(i)), '*.png')))
        path_dict['{}'.format(int(i))] += imgs_path
        
    return path_dict


def make_abcd_dataset(source_dict, target_dict, d_list=[5, 6, 7, 8, 9], max_num=5000, cls_flg=False):
    X_a, y_a, X_b, y_b, X_c, y_c, X_d, y_d = [], [], [], [], [], [], [], []
    src_list = list(source_dict.values())
    for i, s in enumerate(src_list):
        if not i in d_list: 
            X_a.extend(s[:max_num])
            y_a.extend([i for _ in range(max_num)])
        else:
            X_b.extend(s[:max_num])
            if cls_flg:
                y_b.extend([i-5 for _ in range(max_num)])
            else:
                y_b.extend([i for _ in range(max_num)])
            
        
    tgt_list = list(target_dict.values())
    for i, t in enumerate(tgt_list):
        if not i in d_list: 
            X_c.extend(t[:max_num])
            y_c.extend([i for _ in range(max_num)])
        else:
            X_d.extend(t[:max_num])
            if cls_flg:
                y_d.extend([i-5 for _ in range(max_num)])
            else:
                y_d.extend([i for _ in range(max_num)])
            
    return (np.asarray(X_a), np.asarray(y_a)), (np.asarray(X_b), np.asarray(y_b)), (np.asarray(X_c), np.asarray(y_c)), (np.asarray(X_d), np.asarray(y_d))


class SingleDataset(Dataset):
    def __init__(self, path, label, transform):
        assert len(path) == len(label)
        self.image_path = path
        self.label = torch.LongTensor(label)
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.image_path[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label[index]

    def __len__(self):
        return len(self.image_path)


class PairDataset(Dataset):
    """Combined source/target dataset for training using d-SNE.
    Attributes
    ----------
    X : dict of PyTorch Tensors (N, H, W, C)
        Images corresponding to samples of source/target datasets.
    y : dict of PyTorch Tensors (N, 1)
        Labels corresponding to samples of source/target datasets.
    intra_idxs : List of pairs of ints
        Indices for pairs of source/target samples w/ matching labels.
    inter_idxs : List of pairs of ints
        Indices for pairs of source/target samples w/o matching labels.
    full_idxs: List of pairs of ints
        Indexes for pairs of source/target samples.
    transform : Compose transform containing PyTorch transforms
        Pre-processing operations to apply to images when calling
        __getitem__.
    Methods
    -------
    __len__
        Reflect amount of available pairs of indices.
    __getitem__
        Get pair of source and target images/labels.
    Notes
    -----
    d-SNE trains networks using two datasets simultaneously. Of note,
    with d-SNE's training procedure, the loss calculation differs for
    intraclass pairs (y_src == y_tgt) versus interclass pairs
    (y_src != y_tgt).
    By pre-determining pairs of images using a paired dataset, the ratio
    of intraclass and interclass pairs can be controlled. This would be
    more difficult to manage if images were sampled separately from each
    dataset.
    """

    def __init__(self, src_path, src_label, tgt_path, tgt_label, src_num=-1, tgt_num=10,
                 sample_ratio=3, transform=()):
        
        super().__init__()
        self.transform = transform

        # Sample datasets using configuration parameters
        self.X, self.y = {}, {}
        self.X['src'], self.y['src'] = self._resample_data(src_path, src_label,
                                                           src_num)
        self.X['tgt'], self.y['tgt'] = self._resample_data(tgt_path, tgt_label,
                                                           tgt_num)
        self.intra_idxs, self.inter_idxs = self._create_pairs(sample_ratio)
        self.full_idxs = np.concatenate((self.intra_idxs, self.inter_idxs))

        # Sort as to allow shuffling to be performed by the DataLoader
        self.full_idxs = self.full_idxs[np.lexsort((self.full_idxs[:, 1],
                                                    self.full_idxs[:, 0]))]

    def _resample_data(self, X, y, N):
        """Limit sampling to N instances per class."""
        if N > 0:
            # Split labels into set of indexes for each class
            class_idxs = [np.where(y == c)[0] for c in np.unique(y)]

            # Shuffle each of sets of indexes
            [np.random.shuffle(i) for i in class_idxs]

            # Take N indexes, or fewer if total is less than N
            subset_idx = [i[:N] if len(i) >= N else i for i in class_idxs]

            # Use advanced indexing to get subsets of X and y
            idxs = np.array(subset_idx).ravel()
            np.random.shuffle(idxs)
            X, y = X[idxs], y[idxs]

        return X, y

    def _create_pairs(self, sample_ratio):
        """Enforce ratio of inter/intraclass pairs of samples."""
        # Broadcast target/source labels into mesh grid
        # `source` -> (N, 1) broadcast to (N, M)
        # `target` -> (1, M) broadcast to (N, M)
        target, source = np.meshgrid(self.y['tgt'], self.y['src'])

        # Find index pairs (i_S, i_T) for where src_y == tgt_y
        intra_pair_idxs = np.argwhere(source == target)

        # Find index pairs (i_S, i_T) for where src_y != tgt_y
        inter_pair_idxs = np.argwhere(source != target)

        # Randomly sample the interclass pairs to meet desired ratio
        if sample_ratio > 0:
            n_interclass = sample_ratio * len(intra_pair_idxs)
            np.random.shuffle(inter_pair_idxs)
            inter_pair_idxs = inter_pair_idxs[:n_interclass]

            # Sort as to allow shuffling to be performed by the DataLoader
            inter_pair_idxs = inter_pair_idxs[
                np.lexsort((inter_pair_idxs[:, 1], inter_pair_idxs[:, 0]))
            ]

        return intra_pair_idxs, inter_pair_idxs

    def __len__(self):
        """Reflect amount of available pairs of indices."""
        return len(self.full_idxs)

    def __getitem__(self, idx):
        """Get pair of source and target images/labels."""
        src_idx, tgt_idx = self.full_idxs[idx]

        X = {'src': self.X['src'][src_idx], 'tgt': self.X['tgt'][tgt_idx]}
        for key, value in X.items():
            path = X[key]
            image = Image.open(path).convert("RGB")
            X[key] = self.transform(image)

        y = {'src': self.y['src'][src_idx], 'tgt': self.y['tgt'][tgt_idx]}

        return X, y


class SingleDataLoader(Dataset):
    def __init__(self, dataset, shuffle=True, batch_size=1):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.dataset))
        else:
            self.order = np.arange(len(self.dataset))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        X_list, y_list = [], []
        for i in range(self.batch_size):
            X, y = self.dataset[jdx[i]]
            
            X_list.append(X)
            y_list.append(y)

        return torch.stack(X_list), torch.stack(y_list)

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.dataset))

        
class PairDataLoader(Dataset):
    def __init__(self, dataset1, dataset2, shuffle=True, batch_size=1):
        super().__init__()

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.dataset1))
        else:
            self.order = np.arange(len(self.dataset1))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.dataset1)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        X_list1, y_list1 = [], []
        X_list2, y_list2 = [], []
        for i in range(self.batch_size):
            X1, y1 = self.dataset1[jdx[i]]
            X2, y2 = self.dataset2[jdx[i]]
            
            X_list1.append(X1)
            y_list1.append(y1)
            X_list2.append(X2)
            y_list2.append(y2)

        return (torch.stack(X_list1), torch.stack(y_list1)), (torch.stack(X_list2), torch.stack(y_list2))

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.dataset1))
