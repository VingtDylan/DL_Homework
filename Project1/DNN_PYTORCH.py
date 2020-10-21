import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD as optimSGD
from torch.nn import CrossEntropyLoss as CEL

from matplotlib import pyplot as plt 

from MyUtil import *

class DNN_PYTORCH(torch.nn.Module):
    def __init__(self, layer = [4,10,20,3], weight = None):
        super(DNN_PYTORCH, self).__init__()
        self.hidden1 = torch.nn.Linear(layer[0], layer[1])
        self.hidden2 = torch.nn.Linear(layer[1], layer[2])
        self.out = torch.nn.Linear(layer[2], layer[3])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad_(False)
        
        if weight is not None:
            self.hidden1.weight.data = Variable(torch.Tensor(weight[0]).float())
            self.hidden2.weight.data = Variable(torch.Tensor(weight[1]).float())
            self.out.weight.data = Variable(torch.Tensor(weight[2]).float())

        # for name, param in self.named_parameters():
	    #     print(name, param)

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.out(x)
        return x

    def show_grad(self):
        print("pytorch自动求梯度")
        print("网络层梯度按照从输入层到输出层的顺序依次为:\n")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(np.around(m.weight.grad.numpy(),decimals = 6))
                print("\n")
        # for name, param in self.named_parameters():
        #     print(name, param.grad)

    def train_validate(self,X, Y, Vx, Vy, Epochs = 300, batch = 12, lr = 0.05, optim = optimSGD, criterion = CEL,show = True, grad_show = False)->None:
        tensor_train_x = Variable(torch.Tensor(X).float())
        tensor_train_y = Variable(torch.Tensor(Y).long())
        tensor_validate_x = Variable(torch.Tensor(Vx).float())
        tensor_validate_y = Variable(torch.Tensor(Vy).long())
        TrainEpoch = np.linspace(1, Epochs, Epochs)
        ValidateEpoch = np.linspace(1, Epochs, Epochs)
        TrainLoss = []
        ValidateLoss = []
        times = int(len(Y) / batch)
        loss = None
        for epoch in range(Epochs):
            for i in range(times):
                if batch > 1:
                    out = self(tensor_train_x[batch * i : batch * i + batch - 1])
                    _, targets = tensor_train_y[batch * i : batch * i + batch - 1].max(dim = -1)
                    train_loss = criterion(out, targets)
                    optim.zero_grad()   
                    train_loss.backward()  
                else:
                    out = self(tensor_train_x[i]).unsqueeze(0)  
                    _, targets = tensor_train_y[i].max(dim = -1)
                    targets = targets.unsqueeze(0)
                    train_loss = criterion(out, targets)
                    optim.zero_grad()   
                    train_loss.backward()  
                    if grad_show:
                        self.show_grad()
                        break
                optim.step()
                # print('number of epoch : {} , index : {} , loss : {}'.format(epoch,i,train_loss.data))
            out = self(tensor_validate_x)
            _, targets = tensor_validate_y.max(dim = -1)
            validate_loss = criterion(out, targets)
            TrainLoss.append(train_loss.data)
            ValidateLoss.append(validate_loss.data)
        if show:
            plt.figure(2)
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            # Train Loss Curve
            plt.sca(ax1)
            plt.title("Train Loss") 
            plt.xlabel("epoch") 
            plt.ylabel("loss") 
            plt.plot(TrainEpoch,TrainLoss) 
            # Validate Loss Curve
            plt.sca(ax2)
            plt.title("Validate Loss") 
            plt.xlabel("epoch") 
            plt.ylabel("loss") 
            plt.plot(ValidateEpoch,ValidateLoss) 
            plt.show()
    
    def test(self, X, Y)->None:
        tensor_test_x = Variable(torch.Tensor(X).float())
        tensor_test_y = Variable(torch.Tensor(Y).long())
        confusion_matrix = np.zeros(shape = (3,3), dtype = int)
        cases = len(X)
        for i in range(cases):
            out = self(tensor_test_x[i]).unsqueeze(0)
            _, y = tensor_test_y[i].max(dim = -1)
            x = torch.max(out,1)[1]
            confusion_matrix[x.item()][y.item()] += 1
        s = np.sum(np.diagonal(confusion_matrix))
        print("准确率 : {0:.2f}%".format(s / cases * 100))

def main()->None:
    # 随机种子等固定
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_path = "iris.data"
    iris_features, iris_class = load_data(file_path)
    train_x , train_y , validate_x , validate_y , test_x , test_y = train_test_validate_split(iris_features, iris_class, ratio = [0.8,0.1,0.1],random_state = 0)

    epochs = 400
    lr = 0.05
    # 定义layer和weight
    layer = [4,10,20,3]
    weight12 = np.random.normal(loc=0., scale=1., size=(layer[1], layer[0])) / np.sqrt(layer[0])
    weight23 = np.random.normal(loc=0., scale=1., size=(layer[2], layer[1])) 
    weight34 = np.random.normal(loc=0., scale=1., size=(layer[3], layer[2])) 
    weight = [weight12,weight23,weight34]
    
    net = DNN_PYTORCH(layer = layer,weight = weight)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr) 
    criterion = CEL()
    net.train_validate(train_x, train_y, validate_x , validate_y, Epochs = epochs, batch = 12, lr = 0.05, optim = optimizer, criterion = criterion, show = True, grad_show = True)
    net.test(test_x,test_y)

if __name__ == '__main__':
    main()