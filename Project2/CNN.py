import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from Layer import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #  Conv2D 1
        self.conv1 = MyConv2d(1, 6, 5)
        self.pool1 = MyAvgPool2d(2)
        self.conv_fc1 = MyTanh()
        #  Conv2D 2
        self.conv2 = MyConv2d(6, 16, 5)
        self.pool2 = MyAvgPool2d(2)
        self.conv_fc2 = MyTanh()
        # Fully Connnected Layer
        self.fc1 = MyLinear(256, 120)
        self.active_fc1 = MyTanh()
        self.fc2 = MyLinear(120, 84)
        self.active_fc2 = MyTanh()
        self.fc3 = MyLinear(84, 10)
        self.active_fc3 = MyTanh()

    def forward(self, x):
        #  Conv2D 1
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv_fc1(y)
        #  Conv2D 2
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.conv_fc2(y)
        # Fully Connnected Layer
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.active_fc1(y)
        y = self.fc2(y)
        y = self.active_fc2(y)
        y = self.fc3(y)
        y = self.active_fc3(y)
        return y
    
    def loss(self, device: torch.device = torch.device('cpu'), dataLoader : DataLoader = None, criterion = CrossEntropyLoss()) -> float:
        _lossSum, _sum = 0, 0           
        for _, (x, label) in enumerate(dataLoader):
            x = x.to(device)
            label = label.to(device)
            label_np = np.zeros((label.shape[0], 10))
            try:
                predict_y = self(x.float()).detach()
                _error = criterion(predict_y, label.long()).cpu()
                _s = x.shape[0]
                _lossSum  = _lossSum + _error * _s
                _sum = _sum + _s
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    # print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception                
        return _lossSum / _sum

    def train(self, device: torch.device = torch.device('cpu'), train_loader: DataLoader = None, test_loader: DataLoader = None, epoch: int = 100, optimizer = SGD, criterion = CrossEntropyLoss()) -> None:
        Epoch = np.linspace(1, epoch, epoch)
        TrainLoss = []
        TestLoss = []
        TrainAcc = []
        TestAcc = []

        for _epoch in range(epoch):
            for _idx, (train_x, train_label) in enumerate(train_loader):
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                label_np = np.zeros((train_label.shape[0], 10))
                optimizer.zero_grad()
                try:
                    predict_y = self(train_x.float())
                    _error = criterion(predict_y, train_label.long()).cpu()
                    if _idx % 100 == 0:
                        print('epoch: {}, idx: {}, error: {}'.format(_epoch + 1, _idx, _error))
                    _error.backward()
                    optimizer.step()  
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        # print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception     
            TrainLoss.append(self.loss(device,train_loader))
            TestLoss.append(self.loss(device,test_loader))
            TrainAcc.append(self.accuracy(device,train_loader))
            TestAcc.append(self.accuracy(device,test_loader))

        plt.figure(1)
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        # Train Loss Curve
        plt.sca(ax1)
        plt.title("Train Loss") 
        plt.xlabel("epoch") 
        plt.ylabel("loss") 
        plt.plot(Epoch,TrainLoss) 
        # Test Loss Curve
        plt.sca(ax2)
        plt.title("Test Loss") 
        plt.xlabel("epoch") 
        plt.ylabel("loss") 
        plt.plot(Epoch,TestLoss) 
        plt.show()

        plt.figure(2)
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        # Train Acc Curve
        plt.sca(ax1)
        plt.title("Train Acc") 
        plt.xlabel("epoch") 
        plt.ylabel("Acc") 
        plt.plot(Epoch,TrainAcc) 
        # Test Acc Curve
        plt.sca(ax2)
        plt.title("Test Acc") 
        plt.xlabel("epoch") 
        plt.ylabel("Acc") 
        plt.plot(Epoch,TestAcc) 
        plt.show()
            
    def accuracy(self, device: torch.device = torch.device('cpu'), dataLoader: DataLoader = None) -> float:
        correct = 0
        _sum = 0
        for _, (x, label) in enumerate(dataLoader):
            try:
                predict_y = self(x.to(device).float()).detach().cpu()
                predict_ys = np.argmax(predict_y, axis=-1)
                label_np = label.numpy()
                _ = predict_ys == label
                correct += np.sum(_.numpy(), axis=-1)
                _sum += _.shape[0]
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    # print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception  
        return round(correct / _sum * 100, 2)

    def test(self, device: torch.device = torch.device('cpu'), train_loader: DataLoader = None ,test_loader: DataLoader = None) -> None:
        acc1 = self.accuracy(device, train_loader)
        print('训练集 accuracy: {:.2f}%'.format(acc1))
        acc2 = self.accuracy(device, test_loader)
        print('测试集 accuracy: {:.2f}%'.format(acc2))

    def save(self, path = "models/mnist.pkl"):
        torch.save(self, 'models/mnist1.pkl')

def main():
    # 随机种子等固定
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    batch_size = 256 # 128
    train_dataset = mnist.MNIST(root = './train', train = True, transform = ToTensor(), download = True)
    test_dataset = mnist.MNIST(root = './test', train = False, transform = ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    model = CNN().to(device)
    epoch = 30
    lr = 0.001

    # optimizer = SGD(model.parameters(), lr = lr)
    optimizer = Adam(model.parameters(), lr = lr ,betas = (0.9, 0.999), eps = 1e-6)
    criterion = CrossEntropyLoss().to(device)

    model.train(device, train_loader, test_loader, epoch, optimizer, criterion)
    model.test(device, train_loader, test_loader)
    model.save()

if __name__ == '__main__':
    main()