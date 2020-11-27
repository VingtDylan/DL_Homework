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
        # parameter
        self.weight_ih, self.weight_hh, self.bias_hh, self.bias_ih = [], [], [], []
        # input layer 
        self.weight_ih_l0 = Parameter(Tensor(4 * self.hidden_size, self.num_direction * self.input_size))
        self.weight_hh_l0 = Parameter(Tensor(4 * self.hidden_size, self.hidden_size))
        self.bias_ih_l0 = Parameter(Tensor(4 * hidden_size))
        self.bias_hh_l0 = Parameter(Tensor(4 * hidden_size))
        self.weight_ih.append(self.weight_ih_l0)
        self.weight_hh.append(self.weight_hh_l0)
        self.bias_hh.append(self.bias_ih_l0)
        self.bias_ih.append(self.bias_hh_l0)
        # hidden layer
        for i in range(1, num_layers):
            weight_ih_li = Parameter(Tensor(4 * self.hidden_size, self.num_direction * self.hidden_size))
            weight_hh_li = Parameter(Tensor(4 * self.hidden_size, self.hidden_size))
            bias_ih_li = Parameter(Tensor(4 * self.hidden_size))
            bias_hh_li = Parameter(Tensor(4 * self.hidden_size))
            self.register_parameter('weight_ih_l' + str(i) , weight_ih_li)
            self.register_parameter('weight_hh_l' + str(i) , weight_hh_li)
            self.register_parameter('bias_ih_l' + str(i) , bias_ih_li)
            self.register_parameter('bias_hh_l' + str(i) , bias_hh_li)
            self.weight_ih.append(weight_ih_li)
            self.weight_hh.append(weight_hh_li)
            self.bias_ih.append(bias_ih_li)
            self.bias_hh.append(bias_hh_li)

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

        for seq in range(seq_size):
            x_t = input[:, seq, :].t()
            i, f, g, o = self.split["i"], self.split["f"], self.split["g"], self.split["o"]

            for tp in range(self.num_layers):
                h_tp = h_t[tp,:,:].t()
                c_tp = c_t[tp,:,:].t()

                i_t = torch.sigmoid(self.weight_ih[tp][i] @ x_t + self.bias_ih[tp][i].unsqueeze(0).t() + self.weight_hh[tp][i] @ h_tp + self.bias_hh[tp][i].unsqueeze(0).t())
                f_t = torch.sigmoid(self.weight_ih[tp][f] @ x_t + self.bias_ih[tp][f].unsqueeze(0).t() + self.weight_hh[tp][f] @ h_tp + self.bias_hh[tp][f].unsqueeze(0).t())
                g_t =    torch.tanh(self.weight_ih[tp][g] @ x_t + self.bias_ih[tp][g].unsqueeze(0).t() + self.weight_hh[tp][g] @ h_tp + self.bias_hh[tp][g].unsqueeze(0).t())
                o_t = torch.sigmoid(self.weight_ih[tp][o] @ x_t + self.bias_ih[tp][o].unsqueeze(0).t() + self.weight_hh[tp][o] @ h_tp + self.bias_hh[tp][o].unsqueeze(0).t())
                
                c_tp = f_t * c_tp + i_t * g_t
                h_tp = o_t * torch.tanh(c_tp)

                x_t  = h_tp

                c_tp_next = c_tp.t().unsqueeze(0)
                h_tp_next = h_tp.t().unsqueeze(0)    
                h_t[tp,:,:] = h_tp_next
                c_t[tp,:,:] = c_tp_next
            hidden_seq.append(h_tp_next)
        hidden_seq_p = torch.cat(hidden_seq, dim=0)
        hidden_seq_next = torch.transpose(hidden_seq_p, 0, 1)
        return hidden_seq_next, (h_t, c_t)

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

    input = torch.randn(5, 3, 2)
    h0 = torch.randn(2, 5, 3)
    c0 = torch.randn(2, 5, 3)
    rnn = nn.LSTM(input_size = 2, hidden_size = 3, num_layers = 2, batch_first = True, dropout = 0)
    print("LSTM库的输出")
    reset_weigths(rnn)
    # show_weight(rnn)
    output, (hn, cn) = rnn(input, (h0, c0))
    print("LSTM->output输出如下")
    print(output.detach().numpy())
    # print("LSTM->hn输出如下")
    # print(hn.detach().numpy())
    # print("LSTM->cn输出如下")
    print(cn.detach().numpy())
    # print(hn.shape)
    # print(cn.shape)
    
    print("\n")

    myrnn = MyLSTM(input_size = 2, hidden_size = 3, num_layers = 2, batch_first = True, dropout = 0)
    print("自己实现的MyLSTM类的输出")
    reset_weigths(myrnn)
    # show_weight(myrnn)
    myoutput, (myhn, mycn) = myrnn(input, (h0, c0))
    print("MyLSTM->output输出如下")
    print(myoutput.detach().numpy())
    # print("MyLSTM->hn输出如下")
    # print(myhn.detach().numpy())
    # print("MyLSTM->cn输出如下")
    print(mycn.detach().numpy())
    # print(myhn.shape)
    # print(mycn.shape)

if  __name__ == "__main__":
    main()