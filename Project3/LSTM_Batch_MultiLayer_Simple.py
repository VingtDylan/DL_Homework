import random
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter

from Util import *

class MyLSTM(nn.Module):    
    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.num_direction = 1
        # parameter
        self.weight_ii, self.weight_if, self.weight_ig, self.weight_io = [], [], [], []
        self.weight_hi, self.weight_hf, self.weight_hg, self.weight_ho = [], [], [], [] 
        self.bias_ii, self.bias_if, self.bias_ig, self.bias_io = [], [], [], []
        self.bias_hi, self.bias_hf, self.bias_hg, self.bias_ho = [], [], [], [] 
        # input layer 
        self.weight_i0_i = Parameter(Tensor(self.hidden_size, self.num_direction * self.input_size))
        self.weight_i0_f = Parameter(Tensor(self.hidden_size, self.num_direction * self.input_size))
        self.weight_i0_g = Parameter(Tensor(self.hidden_size, self.num_direction * self.input_size))
        self.weight_i0_o = Parameter(Tensor(self.hidden_size, self.num_direction * self.input_size))
        self.weight_ii.append(self.weight_i0_i)
        self.weight_if.append(self.weight_i0_f)
        self.weight_ig.append(self.weight_i0_g)
        self.weight_io.append(self.weight_i0_o)

        self.weight_h0_i = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.weight_h0_f = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.weight_h0_g = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.weight_h0_o = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.weight_hi.append(self.weight_h0_i)
        self.weight_hf.append(self.weight_h0_f)
        self.weight_hg.append(self.weight_h0_g)
        self.weight_ho.append(self.weight_h0_o)

        self.bias_i0_i = Parameter(Tensor(self.hidden_size))
        self.bias_i0_f = Parameter(Tensor(self.hidden_size))
        self.bias_i0_g = Parameter(Tensor(self.hidden_size))
        self.bias_i0_o = Parameter(Tensor(self.hidden_size))
        self.bias_ii.append(self.bias_i0_i)
        self.bias_if.append(self.bias_i0_f)
        self.bias_ig.append(self.bias_i0_g)
        self.bias_io.append(self.bias_i0_o)

        self.bias_h0_i = Parameter(Tensor(self.hidden_size))
        self.bias_h0_f = Parameter(Tensor(self.hidden_size))
        self.bias_h0_g = Parameter(Tensor(self.hidden_size))
        self.bias_h0_o = Parameter(Tensor(self.hidden_size))
        self.bias_hi.append(self.bias_h0_i)
        self.bias_hf.append(self.bias_h0_f)
        self.bias_hg.append(self.bias_h0_g)
        self.bias_ho.append(self.bias_h0_o)
        
        # hidden layer
        for i in range(1, num_layers):
            weight_ii_i = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_ii_f = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_ii_g = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_ii_o = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            self.register_parameter('weight_i' + str(i) + '_i' , weight_ii_i)
            self.register_parameter('weight_i' + str(i) + '_f' , weight_ii_f)
            self.register_parameter('weight_i' + str(i) + '_g' , weight_ii_g)
            self.register_parameter('weight_i' + str(i) + '_o' , weight_ii_o)
            self.weight_ii.append(weight_ii_i)
            self.weight_if.append(weight_ii_f)
            self.weight_ig.append(weight_ii_g)
            self.weight_io.append(weight_ii_o)

            weight_hi_i = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_hi_f = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_hi_g = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            weight_hi_o = Parameter(Tensor(self.hidden_size, self.num_direction * self.hidden_size))
            self.register_parameter('weight_h' + str(i) + '_i' , weight_hi_i)
            self.register_parameter('weight_h' + str(i) + '_f' , weight_hi_f)
            self.register_parameter('weight_h' + str(i) + '_g' , weight_hi_g)
            self.register_parameter('weight_h' + str(i) + '_o' , weight_hi_o)
            self.weight_hi.append(weight_hi_i)
            self.weight_hf.append(weight_hi_f)
            self.weight_hg.append(weight_hi_g)
            self.weight_ho.append(weight_hi_o)

            
            bias_ii_i = Parameter(Tensor(self.hidden_size))
            bias_ii_f = Parameter(Tensor(self.hidden_size))
            bias_ii_g = Parameter(Tensor(self.hidden_size))
            bias_ii_o = Parameter(Tensor(self.hidden_size))
            self.register_parameter('bias_i' + str(i) + '_i', bias_ii_i)
            self.register_parameter('bias_i' + str(i) + '_f', bias_ii_f)
            self.register_parameter('bias_i' + str(i) + '_g', bias_ii_g)
            self.register_parameter('bias_i' + str(i) + '_o', bias_ii_o)
            self.bias_ii.append(bias_ii_i)
            self.bias_if.append(bias_ii_f)
            self.bias_ig.append(bias_ii_g)
            self.bias_io.append(bias_ii_o)

            bias_hi_i = Parameter(Tensor(self.hidden_size))
            bias_hi_f = Parameter(Tensor(self.hidden_size))
            bias_hi_g = Parameter(Tensor(self.hidden_size))
            bias_hi_o = Parameter(Tensor(self.hidden_size))
            self.register_parameter('bias_h' + str(i) + '_i', bias_hi_i)
            self.register_parameter('bias_h' + str(i) + '_f', bias_hi_f)
            self.register_parameter('bias_h' + str(i) + '_g', bias_hi_g)
            self.register_parameter('bias_h' + str(i) + '_o', bias_hi_o)
            self.bias_hi.append(bias_hi_i)
            self.bias_hf.append(bias_hi_f)
            self.bias_hg.append(bias_hi_g)
            self.bias_ho.append(bias_hi_o)

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
            # print(hx)

            # print(h_t)

        hidden_seq = []

        for seq in range(seq_size):
            x_t = input[:, seq, :].t()
            for tp in range(self.num_layers):
                h_tp = h_t[tp,:,:].t().clone()
                c_tp = c_t[tp,:,:].t().clone()
 
                i_t = torch.sigmoid(self.weight_ii[tp] @ x_t + self.bias_ii[tp].unsqueeze(0).t() + self.weight_hi[tp] @ h_tp + self.bias_hi[tp].unsqueeze(0).t())
                f_t = torch.sigmoid(self.weight_if[tp] @ x_t + self.bias_if[tp].unsqueeze(0).t() + self.weight_hf[tp] @ h_tp + self.bias_hf[tp].unsqueeze(0).t())
                g_t =    torch.tanh(self.weight_ig[tp] @ x_t + self.bias_ig[tp].unsqueeze(0).t() + self.weight_hg[tp] @ h_tp + self.bias_hg[tp].unsqueeze(0).t())
                o_t = torch.sigmoid(self.weight_io[tp] @ x_t + self.bias_io[tp].unsqueeze(0).t() + self.weight_ho[tp] @ h_tp + self.bias_ho[tp].unsqueeze(0).t())

                c_tp = f_t * c_tp + i_t * g_t
                h_tp = o_t * torch.tanh(c_tp)
                
                x_t = h_tp
            
                c_tp = c_tp.t().unsqueeze(0)    
                h_tp = h_tp.t().unsqueeze(0)    
                h_t[tp,:,:] = h_tp
                c_t[tp,:,:] = c_tp
            hidden_seq.append(h_tp)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = torch.transpose(hidden_seq, 0, 1)
        # hidden_seq = torch.transpose(hidden_seq, 0, 1)

        return hidden_seq, (h_t, c_t)

def reset_weigths(model):
    for weight in model.parameters():
        init.constant_(weight, 0.5)

def show_weight(model):
    for param in model.named_parameters():
        print(param)

def main():
    set_seed(10)
    input = torch.randn(5, 3, 4)
    h0 = torch.randn(2, 5, 3)
    c0 = torch.randn(2, 5, 3)
    rnn = nn.LSTM(input_size = 4, hidden_size = 3, num_layers = 2, batch_first = True)
    print("LSTM库的输出")
    reset_weigths(rnn)
    # show_weight(rnn)
    output, (hn, cn) = rnn(input, (h0, c0))
    print("LSTM->output输出如下")
    print(output.detach().numpy())
    # print("LSTM->hn输出如下")
    # print(hn.detach().numpy())
    # print("LSTM->cn输出如下")
    # print(cn.detach().numpy())
    # print(hn.shape)
    # print(cn.shape)

    print("\n")

    myrnn = MyLSTM(input_size = 4, hidden_size = 3, num_layers = 2, batch_first = True)
    print("自己实现的MyLSTM类的输出")
    reset_weigths(myrnn)
    # show_weight(myrnn)
    myoutput, (myhn, mycn) = myrnn(input, (h0, c0))
    print("MyLSTM->output输出如下")
    print(myoutput.detach().numpy())
    # print("MyLSTM->hn输出如下")
    # print(myhn.detach().numpy())
    # print("MyLSTM->cn输出如下")
    # print(mycn.detach().numpy())
    # print(myhn.shape)
    # print(mycn.shape)


if  __name__ == "__main__":
    main()