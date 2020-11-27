import torch
import torch.nn as nn
from torch.autograd import Variable

from LSTM_Batch_MultiLayer import *

class Stock3(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , batch_first=True):
        super(Stock3, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.rnn1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.rnn2 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.rnn3 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.rnn4 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.rnn5 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
        self.rnn6 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first)
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

        out1, (hn1, cn1) = self.rnn1(x, (h0,c0))
        out2, (hn2, cn2) = self.rnn2(x, (h0,c0))
        out3, (hn3, cn3) = self.rnn3(x, (h0,c0))
        out4, (hn4, cn4) = self.rnn4(x, (h0,c0))
        out5, (hn5, cn5) = self.rnn5(x, (h0,c0))
        out6, (hn6, cn6) = self.rnn6(x, (h0,c0))

        out1 = self.linear1(out1[:, -1, :]) 
        out2 = self.linear2(out2[:, -1, :]) 
        out3 = self.linear3(out3[:, -1, :]) 
        out4 = self.linear4(out4[:, -1, :]) 
        out5 = self.linear5(out5[:, -1, :]) 
        out6 = self.linear6(out6[:, -1, :]) 
        return [out1, out2, out3, out4, out5, out6]
    
    def train(self, args, train_loader, criterion, optimizer):
        for i in range(args.epochs):
            for idx, (data, label) in enumerate(train_loader):
                if args.useGPU:
                    data = data.squeeze(1).cuda()
                    pred = self(Variable(data).cuda())
                    label = label.cuda()
                else:
                    data = data.squeeze(1)
                    pred = self(Variable(data))
                loss = criterion(pred, label.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(i, loss.item())
    
    def train_test(self, args, train_loader):
        s = 0
        t = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = self(Variable(data).cuda())
                _, predicted = torch.max(pred.data, 1)
                t += label.size(0)
                s += (predicted.cpu() == label.cpu()).sum()
            else:
                data = data.squeeze(1)
                pred = self(Variable(data))
                _, predicted = torch.max(pred.data, 1)
                t += label.size(0)
                s += (predicted == label).sum()
        print("训练集样本个数: " + str(t))
        print("训练集正确预测个数: " + str(s.item()))
        print("训练集准确率: " + str(round((s * 100.0 / t).item(), 2)))
        print("----------------------------1")
    
    def test_test(self, args, test_loader, file_name):
        s = 0
        t = 0
        for idx, (data, label) in enumerate(test_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = self(Variable(data).cuda())
                _, predicted = torch.max(pred.data, 1)
                t += label.size(0)
                s += (predicted.cpu() == label.cpu()).sum()
            else:
                data = data.squeeze(1)
                pred = self(Variable(data))
                _, predicted = torch.max(pred.data, 1)
                t += label.size(0)
                s += (predicted == label).sum()
        print("测试集样本个数: " + str(t))
        print("测试集正确预测个数: " + str(s.item()))
        print("测试集准确率: " + str(round((s * 100.0 / t).item(), 2)))
        print("----------------------------2")
        acc = s * 100.0 / t
        acc = acc.item()
        print("base : 53.16; baseline: 55.01")
        if(acc > 55.01):
            torch.save(self, "models/" + file_name +"_acc_" + str(round(acc, 2)) + ".pkl")
            print("model with acc " + str(round(acc, 2)) + "%" + " is saved!")