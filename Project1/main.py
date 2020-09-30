import torch
import torch
import torch.nn as nn
import torchvision


def configure():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))

    flag = torch.cuda.is_available()
    if flag:
        print("CUDA可使用")
    else:
        print("CUDA不可用")

    ngpu= 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("驱动为：",device)
    print("GPU型号： ",torch.cuda.get_device_name(0))

if __name__ == '__main__':
    configure()
