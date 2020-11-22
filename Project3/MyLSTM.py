import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True, dropout = 0):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout # drop

        self.num_direction = 1
        # 2 3 1 register_parameter
        self.weight_ih_l0 = Parameter(Tensor(4 * self.hidden_size, self.num_direction * self.input_size))
        self.weight_hh_l0 = Parameter(Tensor(4 * self.hidden_size, self.hidden_size))
        self.bias_ih_l0 = Parameter(Tensor(4 * hidden_size))
        self.bias_hh_l0 = Parameter(Tensor(4 * hidden_size))

        self.split = {}
        self.split["i"] = np.linspace(0, self.hidden_size, self.hidden_size, endpoint = False)
        self.split["f"] = np.linspace(self.hidden_size, 2 * self.hidden_size, self.hidden_size, endpoint = False)
        self.split["g"] = np.linspace(2 * self.hidden_size, 3 * self.hidden_size, self.hidden_size, endpoint = False)
        self.split["o"] = np.linspace(3 * self.hidden_size, 4 * self.hidden_size, self.hidden_size, endpoint = False)

        self.reset_weigths()
    def reset_weigths(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx = None): 
        assert(self.batch_first == True)
        batch_size, seq_size , features_size = input.size()
        if hx is None:
            h_t = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
            c_t = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
        else:
            (h_t, c_t) = hx

        hidden_seq = []

        for t in range(seq_size):
            x_t = input[:, t, :].t()
            h_t = h_t[0,:,:].t()
            c_t = c_t[0,:,:].t()

            i, f, g, o = self.split["i"], self.split["f"], self.split["g"], self.split["o"]

            i_t = torch.sigmoid(self.weight_ih_l0[i] @ x_t + self.bias_ih_l0[i].unsqueeze(0).t() + self.weight_hh_l0[i] @ h_t + self.bias_hh_l0[i].unsqueeze(0).t())
            f_t = torch.sigmoid(self.weight_ih_l0[f] @ x_t + self.bias_ih_l0[f].unsqueeze(0).t() + self.weight_hh_l0[f] @ h_t + self.bias_hh_l0[f].unsqueeze(0).t())
            g_t =    torch.tanh(self.weight_ih_l0[g] @ x_t + self.bias_ih_l0[g].unsqueeze(0).t() + self.weight_hh_l0[g] @ h_t + self.bias_hh_l0[g].unsqueeze(0).t())
            o_t = torch.sigmoid(self.weight_ih_l0[o] @ x_t + self.bias_ih_l0[o].unsqueeze(0).t() + self.weight_hh_l0[o] @ h_t + self.bias_hh_l0[o].unsqueeze(0).t())
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            c_t = c_t.t().unsqueeze(0)
            h_t = h_t.t().unsqueeze(0)
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, (h_t, c_t)

def reset_weigths(model):
    for weight in model.parameters():
        init.constant_(weight, 0.5)

def show_weight(model):
    for param in model.named_parameters():
        print(param)

def main():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 

    input = torch.ones(1, 1, 2)
    h0 = torch.ones(1, 1, 3)
    c0 = torch.ones(1, 1, 3)
    rnn = nn.LSTM(input_size = 2, hidden_size = 3, num_layers = 1, batch_first = True, dropout = 0)
    print("LSTM库的输出")
    reset_weigths(rnn)
    # show_weight(rnn)
    output, (hn, cn) = rnn(input, (h0, c0))
    # print(output.detach().numpy())
    # print(hn.detach().numpy())
    print(cn.detach().numpy())
    # print(hn.shape)
    print(cn.shape)
    
    # print("\n")

    myrnn = MyLSTM(input_size = 2, hidden_size = 3, num_layers = 1, batch_first = True, dropout = 0)
    print("自己实现的MyLSTM类的输出")
    reset_weigths(myrnn)
    # show_weight(myrnn)
    myinput = torch.ones(1, 1, 2)
    myh0 = torch.ones(1, 1, 3)
    myc0 = torch.ones(1, 1, 3)
    myoutput, (myhn, mycn) = myrnn(myinput, (myh0, myc0))
    # print(myoutput.detach().numpy())
    # print(myhn.detach().numpy())
    print(mycn.detach().numpy())
    # print(myhn.shape)
    print(mycn.shape)

if  __name__ == "__main__":
    main()