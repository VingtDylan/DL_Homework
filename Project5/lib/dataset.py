from torch.utils.data import Dataset



import pandas as pd
from ast import literal_eval
from lib.config import nrms

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import tqdm
import random
import nltk
import torch.nn.functional as F

class MyDataSet(Dataset):
    def __init__(self, clicked_news, candidate_news, label):
        super(MyDataSet, self).__init__()
        self.clicked_news = clicked_news
        self.candidate_news = candidate_news
        self.label = label
    
    def __getitem__(self, index):
        try:
            return self.clicked_news[index], self.candidate_news[index], self.label[index]
        except Exception:
            raise NotImplementedError

    def __len__(self):
        try:
            return len(self.label)
        except Exception:
            raise NotImplementedError