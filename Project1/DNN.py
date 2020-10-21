import random
import math

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt 

from MyUtil import *

class DNN:
    def __init__(self, layer = [4,10,20,3], weight = None):
        """初始化
        :param layer: 从输入层到输出层，每一层的结点个数
        :param weight: 网络权重初始化数值
        """
        self.num_layers = len(layer)
        if(self.num_layers <= 2): raise ValueError("num_layers error")
        self.layer_pools = []
        if not weight:
            weight = [None] * (self.num_layers - 1)
        for i in range(self.num_layers - 2):
            self.layer_pools.append(Layer(input_num = layer[i],  neu_num = layer[i + 1], weight = weight[i], activate_func = 'sigmoid'))
        self.layer_pools.append(Layer(input_num = layer[-2],  neu_num = layer[-1], weight = weight[-1], activate_func = 'softmax'))
        # for lay in self.layer_pools:
        #     print(lay.weight)

    def forward(self, input)->np.matrix:
        """前向传播
        :param input: 输入
        :return : 网络的输出
        """
        self.input = input
        out = input
        for i in range(len(self.layer_pools)):
            out = self.layer_pools[i].forward(out)
        return out

    def step(self)->None:
        """更新权重
        """
        for lay in self.layer_pools:
            lay.weight -= self.lr * lay.partial / self.batch_size

    def cross_entropy(self, out, label)->float:
        """计算交叉熵
        :param out: 网络输出值
        :param label: 训练的目标值
        :return : 交叉熵
        """
        assert out.shape == label.shape
        if out.shape[1] == 1:
            out.reshape(1,-1)
            label.reshape(1,-1)
        batch_size = out.shape[1]
        loss = 0
        for i in range(batch_size):
            loss -= np.sum(np.multiply(label[:,i], np.log(out[:,i] + 1e-7)))
        loss /= batch_size
        return round(loss,7)

    def backProp(self, output, label, lr = 0.06,skip = False)->float:
        """ bp核心
        :param output: 网络输出值
        :param label: 训练的目标值
        :param lr: 学习率
        :param skip: 验证集用于跳过权值更新的flag
        :return : 网路的误差
        """
        self.lr = lr
        self.batch_size = output.shape[1]
        loss = self.cross_entropy(output,label)
        if skip: return loss
        delta = self.backPropLoss(output,label)
        partial = np.dot(delta,self.layer_pools[-1].input.transpose())
        self.layer_pools[-1].partial = partial
        for i, layer in reversed(list(enumerate(self.layer_pools))[:-1]):
            delta = np.dot(self.layer_pools[i + 1].weight.transpose(), delta)
            s = layer.backPropS_i()
            delta = np.multiply(delta, s)
            layer.partial = np.dot(delta, layer.input.transpose())
        return loss

    def backPropLoss(self,output,label)->np.matrix:
        """计算误差向量
        :param out: 网络输出值
        :param label: 训练的目标值
        :return : 误差向量
        """
        return output - label

    def show_grad(self):
        """打印梯度
        """
        print("矩阵实现的BP算法")
        print("网络层梯度按照从输入层到输出层的顺序依次为:\n")
        for layer in self.layer_pools:
            print(np.around(layer.partial,decimals = 6))
            print("\n")

    def train_validate(self,X, Y, Vx, Vy, Epochs = 300, batch = 12, lr = 0.05, show = True, grad_show = False)->None:
        """训练和验证
        :param X: 用于训练的features
        :param Y: 用于训练的目标值
        :param Vx: 用于验证的features
        :param Vy: 用于验证的目标值
        :param Epoch: 迭代次数
        :param bacth: 一次参与训练的样本数
        :param lr: 学习率
        :param show: 是否绘制loss的flag
        :param grad_show: 是否打印梯度的flag
        """
        TrainEpoch = np.linspace(1, Epochs, Epochs)
        ValidateEpoch = np.linspace(1, Epochs, Epochs)
        TrainLoss = []
        ValidateLoss = []
        times = int(len(Y) / batch)
        for epoch in range(Epochs):
            for i in range(times):
                if batch > 1:
                    label_pred = self.forward(X[batch * i : batch * i + batch - 1].transpose())
                    train_loss = self.backProp(label_pred,Y[batch * i : batch * i + batch - 1].transpose(),lr = lr)
                else:
                    label_pred = self.forward(X[i].transpose())
                    train_loss = self.backProp(label_pred,Y[i].transpose(),lr = lr)
                    if grad_show:
                        self.show_grad()
                        break
                self.step()
                # print('number of epoch : {} , index : {} , loss : {}'.format(epoch,i,train_loss))
            label_pred = self.forward(Vx.transpose())
            validate_loss = self.backProp(label_pred,Vy.transpose(), skip = True)
            TrainLoss.append(train_loss)
            ValidateLoss.append(validate_loss)
        if show:
            plt.figure(1)
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

    def test(self, X, Y, strs)->None:
        """ 测试
        :param X: 用于测试的features
        :param Y: 用于测试的目标值
        :param strs: 提示string
        """
        print(strs)
        confusion_matrix = np.zeros(shape = (3,3), dtype = int)
        cases = len(X)
        for i in range(cases):
            label_pred = self.forward(X[i].transpose())
            x = np.argmax(Y[i])
            y = np.argmax(label_pred)
            confusion_matrix[x][y] += 1
        s = np.sum(np.diagonal(confusion_matrix))
        print("准确率 : {0:.2f}%".format(s / cases * 100))
        confusion_matrix_sum_col = np.sum(confusion_matrix, axis=0)
        p_sum = 0
        p_count = 0
        for i in range(3):
            if confusion_matrix_sum_col[i] != 0:
                p_sum += confusion_matrix[i][i] / confusion_matrix_sum_col[i]
                p_count += 1
        P = p_sum / p_count
        print("精确率 : {0:.2f}%".format(P * 100))
        confusion_matrix_sum_row = np.sum(confusion_matrix, axis=1)
        r_sum = 0
        r_count = 0
        for i in range(3):
            if confusion_matrix_sum_row[i] != 0:
                r_sum += confusion_matrix[i][i] / confusion_matrix_sum_row[i]
                r_count += 1
        R = r_sum / r_count
        print("召回率 : {0:.2f}%".format(R * 100))
        def P_R(P,R)->float:
            if math.isclose(P + R, 0, rel_tol=1e-9):
                return 0.0
            else:
                return 2 * P * R / (P + R) * 100
        print("F1值 : {0:.2f}%".format(P_R(P,R)))
        
class Layer:
    def __init__(self, input_num = 4, neu_num = 4, weight = None, activate_func = 'sigmoid'):
        """初始化
        :param input_num: 输入的神经元数
        :param neu_num: 本层的神经元数
        :param weight: 本层的初始化权重
        :param activate_func: 本层的激活函数
        """
        self.neu_num = neu_num
        self.Neu_pools = []
        self.activate_func = activate_func
        for i in range(neu_num):
            self.Neu_pools.append(Neu(input_num = input_num)) 
        for i in range(neu_num):
            if weight is not None:
                for j in range(input_num):
                    self.Neu_pools[i].weight[j] = weight[i][j]
            else:
                self.Neu_pools[i].weight = np.random.normal(loc=0., scale=1., size=(1, input_num)) / np.sqrt(input_num)
        self.weight = np.zeros(shape = (neu_num, input_num))
        for i in range(len(self.Neu_pools)):
            self.weight[i] = self.Neu_pools[i].weight

    def forward(self,input)->np.matrix:
        """前向传递
        :param input:输入
        :return 输出
        """
        self.input = input
        output = np.matmul(self.weight, self.input)
        output = self.sigmoid(output) if self.activate_func == 'sigmoid' else self.softmax(output)
        self.output = output
        return output
    
    def softmax(self, x)->np.matrix:
        """ softmax手动实现
        :param x: 输入
        :return : 计算出的softmax概率矩阵
        """
        exps = np.exp(x - np.max(x, axis = 0))
        return exps / exps.sum(axis = 0)

    def sigmoid(self,input):
        """ sigmoidx手动实现
        :param x: 输入
        :return : 计算出的sigmoid
        """
        return 1.0 / (1.0 + np.exp(-input))

    def backPropS_i(self):
        """ sigmoid导数手动实现
        :param x: 输入(sigmoid输出值)
        :return : 计算出的sigmoid导数
        """
        s = np.multiply(self.output , (1 - self.output))
        return s

class Neu:
    def __init__(self, input_num):
        self.weight = [0] * input_num
        self.bias = 0

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
    
    dnn = DNN(layer = layer,weight = weight)
    dnn.train_validate(train_x, train_y, validate_x , validate_y, Epochs = epochs, batch = 12, lr = 0.05, show = True, grad_show = True)
    dnn.test(train_x, train_y, "Train_data evaluation")
    dnn.test(validate_x, validate_y, "Validate_data evaluation")
    dnn.test(test_x, test_y, "Test_data evalution")

if __name__ == '__main__':
    main()