import random

import numpy as np
import torch
import torch.nn as nn


def main():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    a = [1,2,3,4,5,6,7]
    seq = 5
    t = a[1: 1 + seq]
    print(t)
    m = a[1 + seq]
    print(m)
# accuracy_1	55.01%
# accuracy_20	70.29%
# accuracy_60	77.01%

# 53.16%，63.57%，63.97%

if __name__ == "__main__":
    # main()
    s = str(round(1 * 100.0 / 20,2))
    print(s)