import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm2(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True):
        super(lstm2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, (hn, cn) = self.rnn(x, (h0,c0))
        out = self.linear(out[:, -1, :]) 
        return out