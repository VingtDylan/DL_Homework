import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        # self.linear1 = nn.Linear(self.hidden_size, 64)
        # self.linear2 = nn.Linear(64, 16)
        # self.linear3 = nn.Linear(16, self.output_size)

    def forward(self, x):
        # out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # # a, b, c = hidden.shape
        # # out = self.linear(hidden.reshape(a * b, c))
        # print(hidden.shape)
        # print(cell.shape)
        # print(out.shape)
        # out = self.linear(out[:, -1, :])
        
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        # One time step
        out, (hn, cn) = self.rnn(x, (h0,c0))
        
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 

        # print("hint")
        # print(out.shape)
        out = self.linear(out[:, -1, :]) 
        # print(out.shape)
        # out = self.linear1(out[:, -1, :]) 
        # out = self.linear2(out) 
        # out = self.linear3(out) 
        return out