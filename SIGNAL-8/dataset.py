import os
import torch
import pickle
from torch.utils.data.dataset import Dataset
import numpy as np
from glob import glob
from utils import get_real_imag, split_data
import h5py
from sklearn import preprocessing


def load_data_complex(root, length):
    """
    :param root: data set direction
    :param length: fix the time-series length
    :return:
    """
    name2label = {}
    labels = []
    raw_data = []
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())

    IF_data = []
    for name in name2label.keys():
        IF_data += glob(os.path.join(root, name, '*.txt'))

    for index in IF_data:
        data = []  # save data
        label = []  # save label
        with open(index, 'rb') as file_pi:
            x = pickle.load(file_pi)  # all signal samples in a single file
            for idx in range(len(x)):
                x[idx] = get_real_imag(x[idx][0:length], transpose=False)
            data.extend(x)
            keys = index.split('\\')[1]
            label.extend((np.ones(len(data))*name2label[keys]).astype(np.int).tolist())

        raw_data.extend(data)
        labels.extend(label)

    np.random.seed(116)
    np.random.shuffle(raw_data)
    np.random.seed(116)
    np.random.shuffle(labels)

    return raw_data, labels


# load RadioML I/Q data (a pkl file)
def load_RadioML_data(root, filename, type=None):
    if type == 'pkl':
        with open(os.path.join(root, filename), 'rb') as fo:
            list_data = pickle.load(fo, encoding='bytes')

        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], list_data.keys())))),
                         [1, 0])  # different SNRs and mods
        data = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                data.append(list_data[(mod, snr)])
                for i in range(list_data[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        data = np.vstack(data)  # stack data with specific SNR
        label = encoder(lbl, mods)  # encode labels (not one-hot)

    elif type == 'hdf5':
        f = h5py.File(os.path.join(root, filename), 'r')
        data = list(f['X'])
        label = list(f['Y'])
        # normalization
        # data = normalization(data)

    return data, label


def encoder(labels, classes):
    yy1 = []
    for i in range(len(labels)):
        yy1.append(classes.index(labels[i][0]))

    return yy1


def get_index(lst=None, item=None):
    return [index for (index, value) in enumerate(lst) if value == item]


class Dataset_complex(Dataset):
    def __init__(self, root_dir, mode=None):
        """
        complex radar signals
        :param root_dir: data set direction
        :param mode: 'train' or 'test'
        """
        self.root = root_dir
        self.data = load_data_complex(self.root, length=1000)
        self.data_info, self.label_info = split_data(self.data[0], mode=mode), split_data(self.data[1], mode=mode)
        self.data_info, self.label_info = torch.Tensor(self.data_info), torch.Tensor(self.label_info)

    def __getitem__(self, idx):
        return self.data_info[idx], self.label_info[idx]

    def __len__(self):
        return len(self.label_info)


class Dataset_IQ(Dataset):
    def __init__(self, root_dir, filename, mode=None):
        """
        I/Q radar signals: RadioML data
        :param root_dir: data set direction
        :param rate: the proportion of train sample, default >= 0.6
        """
        self.root = root_dir
        self.data, self.label = load_RadioML_data(self.root, filename=filename, type='hdf5')
        self.data_info, self.label_info = split_data(self.data, mode=mode), split_data(self.label, mode=mode)
        self.data_info, self.label_info = torch.Tensor(self.data_info), torch.Tensor(self.label_info)

    def __getitem__(self, idx):
        return self.data_info[idx], self.label_info[idx]

    def __len__(self):
        return len(self.label_info)


