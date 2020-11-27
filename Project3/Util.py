import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

class Mydataset(Dataset):
    def __init__(self, xx, yy):
        self.x = xx
        self.y = yy

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        return x1, y1

    def __len__(self):
        return len(self.x)

def set_seed(seed):
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def split_data_label(sequence, data_label):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence):
        X.append(np.array(data_label.iloc[i: i + sequence, 1:-1], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence, -1], dtype = np.float32))
    return X, Y

def split_data_label_merge(sequence, data_label, delay = 1):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence - delay + 1):
        X.append(np.array(data_label.iloc[i: i + sequence, 1 : -6], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence + delay - 1, -6 : ], dtype = np.float32))
    return X, Y