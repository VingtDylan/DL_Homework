import torch
import torch.nn as nn
from torch.autograd import Variable

from LSTM_Batch_MultiLayer import *
# from LSTM_Batch_MultiLayer_Simple import *

class LSTM_Stock(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , batch_first=True):
        super(LSTM_Stock, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        # 如果需要使用实现的LSTM，请将self.rnn改由 MyLSTM调用
        self.rnn = MyLSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        # 如果需要使用实现的LSTM，请将self.rnn改由 nn.LSTM调用
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.linear4 = nn.Linear(self.hidden_size, self.output_size)
        self.linear5 = nn.Linear(self.hidden_size, self.output_size)
        self.linear6 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        if torch.cuda.is_available() and x.is_cuda:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out, (hn, cn) = self.rnn(x, (h0,c0))
        out1 = self.linear1(out[:, -1, :]) 
        out2 = self.linear2(out[:, -1, :]) 
        out3 = self.linear3(out[:, -1, :]) 
        out4 = self.linear4(out[:, -1, :]) 
        out5 = self.linear5(out[:, -1, :]) 
        out6 = self.linear6(out[:, -1, :]) 
        return [out1, out2, out3, out4, out5, out6]
    
    def train(self, args, train_loader, val_loader, criterion, optimizer):
        for epoch in range(args.epochs):
            train_loss = 0
            train_counter = 0
            for idx, (data, label) in enumerate(train_loader):
                if args.useGPU:
                    data = data.squeeze(1).cuda()
                    pred = self(Variable(data).cuda())
                    label = label.cuda()
                loss = criterion(pred[0], label[:,0].long())
                for i in range(1, 6):
                    loss += criterion(pred[i], label[:,i].long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_counter += label.size(0)

            val_loss = 0
            val_counter = 0
            for idx, (data, label) in enumerate(val_loader):
                if args.useGPU:
                    data = data.squeeze(1).cuda()
                    pred = self(Variable(data).cuda())
                    label = label.cuda()
                loss = criterion(pred[0], label[:,0].long())
                for i in range(1, 6):
                    loss += criterion(pred[i], label[:,i].long())
                val_loss += loss.item()
                val_counter += label.size(0)
            print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \t'.format(epoch + 1, train_loss / train_counter, val_loss / val_counter))
    
    def train_test(self, args, train_loader):
        s = [0] * 6
        t = [0] * 6 
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = self(Variable(data).cuda())[0].cpu()
                _, predicted = torch.max(pred.data, 1)
                for i in range(6):
                    t[i] += label.size(0)
                    s[i] += (predicted.cpu() == label[:,i].cpu()).sum().item()
        acc = [str(round(s[i] * 100.0 / t[i], 2)) + "%" for i in range(6)]

        print("------------训练集--------------")
        print("金属次序: Copper Aluminium, Lead, Nickel, Tin, Zinc")
        print("训练集样本个数: ")
        print(t[0],t[1],t[2],t[3],t[4],t[5])
        print("训练集正确预测个数: ")
        print(s[0],s[1],s[2],s[3],s[4],s[5])
        print("训练集准确率: ")
        print(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5])
        print("平均准确率:")
        aver = round(sum([s[i] * 100.0 / t[i] for i in range(6)]) / 6, 2)
        print(aver)
    
    def test_test(self, args, test_loader, file_name):
        s = [0] * 6
        t = [0] * 6 
        for idx, (data, label) in enumerate(test_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = self(Variable(data).cuda())[0].cpu()
                _, predicted = torch.max(pred.data, 1)
                # predicted.fill_(0)
                for i in range(6):
                    t[i] += label.size(0)
                    s[i] += (predicted.cpu() == label[:,i].cpu()).sum().item()
        acc = [round(s[i] * 100.0 / t[i], 2) for i in range(6)]

        print("------------测试集--------------")
        print("金属次序: Copper Aluminium, Lead, Nickel, Tin, Zinc")
        print("测试集样本个数: ")
        print(t[0],t[1],t[2],t[3],t[4],t[5])
        print("测试集正确预测个数: ")
        print(s[0],s[1],s[2],s[3],s[4],s[5])
        print("测试集准确率: ")
        print(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5])
        print("平均准确率:")
        aver = round(sum([s[i] * 100.0 / t[i] for i in range(6)]) / 6, 2)
        print(aver)
        
        if file_name == "1d": 
            baseline = 55.01
            print("base: 53.16, baseline: 55.01")
        elif file_name == "20d": 
            baseline = 70.29
            print("base: 63.57, baseline: 70.29")
        else:
            baseline = 77.01
            print("base: 63.97, baseline: 77.01")
        if(aver > baseline):
            torch.save(self, "models/" + file_name +"_acc_" + str(aver) + ".pkl")
            print("model with acc " + str(aver) + "%" + " is saved!")