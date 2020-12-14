import random
import os

import torch
import numpy as np

from lib.parser import args

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

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [args.PAD] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
