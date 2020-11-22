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

        # # i f g o are the input, forget, cell, and output gates,
        # # 输入门的权重矩阵和bias矩阵 i
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # 遗忘门的权重矩阵和bias矩阵 f
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # 输出门的权重矩阵和bias矩阵 o
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))
        
        # cell的的权重矩阵和bias矩阵 g
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))

        self.reset_weigths()
    def reset_weigths(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx = None): 
        # assert(self.batch_first == True)
        # batch_size, seq_size , features_size = input.size()
        # if hx is None:
        #     h_t = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
        #     c_t = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
        # else:
        #     (h_t, c_t) = hx
 
        # if self.batch_first:
            # h_t = torch.transpose(h_t, 2, 1)
        #     c_t = torch.transpose(c_t, 2, 1)

        # h_t, c_t = h_t.t(), c_t.t()

        # hidden_seq = []

        # for t in range(seq_size):
        #     x_t = input[:, t, :].t()
        #     print(input)
        #     print(self.weight_ih_l0[self.split["i"]])
        #     print(x_t)
        #     print(self.weight_ih_l0[self.split["i"]] @ x_t)
        #     print("---------------------------------1")
        #     print(self.bias_ih_l0[self.split["i"]].unsqueeze(0).t())
        #     print(self.weight_ih_l0[self.split["i"]] @ x_t + self.bias_ih_l0[self.split["i"]].unsqueeze(0).t())
        #     s = self.weight_ih_l0[self.split["i"]] @ x_t + self.bias_ih_l0[self.split["i"]].unsqueeze(0).t()
        #     print("---------------------------------2")
        #     print(self.weight_hh_l0[self.split["i"]].unsqueeze(0))
        #     print(h_t)
        #     print(self.weight_hh_l0[self.split["i"]].unsqueeze(0) * h_t)
        #     print("---------------------------------3")
        #     print(self.bias_hh_l0[self.split["i"]])
        #     print("\n")
        #     i_t = torch.sigmoid(self.weight_ih_l0[self.split["i"]] @ x_t + self.bias_ih_l0[self.split["i"]].t() + self.weight_hh_l0[self.split["i"]] * h_t + self.bias_hh_l0[self.split["i"]].t())
        #     f_t = torch.sigmoid(self.weight_ih_l0[self.split["f"]] @ x_t + self.bias_ih_l0[self.split["f"]].t() + self.weight_hh_l0[self.split["f"]] * h_t + self.bias_hh_l0[self.split["f"]].t())
        #     g_t =    torch.tanh(self.weight_ih_l0[self.split["g"]] @ x_t + self.bias_ih_l0[self.split["g"]].t() + self.weight_hh_l0[self.split["g"]] * h_t + self.bias_hh_l0[self.split["g"]].t())
        #     o_t = torch.sigmoid(self.weight_ih_l0[self.split["o"]] @ x_t + self.bias_ih_l0[self.split["o"]].t() + self.weight_hh_l0[self.split["o"]] * h_t + self.bias_hh_l0[self.split["o"]].t())

        #     print()

        #     c_t = f_t * c_t + i_t * g_t
        #     h_t = o_t * torch.tanh(c_t)
        #     hidden_seq.append(h_t)
        #     print(h_t)
        #     print(o_t.shape)


        if hx is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = hx
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        seq_size = 1
        for t in range(seq_size):
            x = input[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)
            print("i_t...")
            # print(self.w_ii)
            # print(x)
            # print(self.w_ii @ x)
            # print(self.b_ii)
            # print(self.w_hi)
            # print(h_t)
            # print(self.w_hi @ h_t)
            # print(self.w_ii @ x + self.b_ii + self.w_hi @ h_t)
            # print(self.b_hi)

            # print(i)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)
            # print(f)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t + self.b_hg)
            # print(g)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +self.b_ho)
            # print(o)
            
            print(f,f.shape)
            print(c_t,c_t.shape)
            print(f * c_t)
            c_next = f * c_t + i * g
            print(c_next)
            h_next = o * torch.tanh(c_next)
            # print(h_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)

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
    
    print("\n")

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

        # # i f g o are the input, forget, cell, and output gates,
        # # 输入门的权重矩阵和bias矩阵 i
        # self.w_ii = Parameter(Tensor(hidden_size, input_size))
        # self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        # self.b_ii = Parameter(Tensor(hidden_size, 1))
        # self.b_hi = Parameter(Tensor(hidden_size, 1))

        # # 遗忘门的权重矩阵和bias矩阵 f
        # self.w_if = Parameter(Tensor(hidden_size, input_size))
        # self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        # self.b_if = Parameter(Tensor(hidden_size, 1))
        # self.b_hf = Parameter(Tensor(hidden_size, 1))

        # # 输出门的权重矩阵和bias矩阵 o
        # self.w_io = Parameter(Tensor(hidden_size, input_size))
        # self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        # self.b_io = Parameter(Tensor(hidden_size, 1))
        # self.b_ho = Parameter(Tensor(hidden_size, 1))
        
        # # cell的的权重矩阵和bias矩阵 g
        # self.w_ig = Parameter(Tensor(hidden_size, input_size))
        # self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        # self.b_ig = Parameter(Tensor(hidden_size, 1))
        # self.b_hg = Parameter(Tensor(hidden_size, 1))

            # print("---------------------------------0")
            # print(self.split["i"])
            # print(self.weight_ih_l0[self.split["i"]])
            # print(x_t)
            # print(self.weight_ih_l0[self.split["i"]] @ x_t)
            # print("---------------------------------1")
            # print(self.bias_ih_l0[self.split["i"]].unsqueeze(0).t())
            # print(self.weight_ih_l0[self.split["i"]] @ x_t + self.bias_ih_l0[self.split["i"]].unsqueeze(0).t())
            # s = self.weight_ih_l0[self.split["i"]] @ x_t + self.bias_ih_l0[self.split["i"]].unsqueeze(0).t()
            # print("---------------------------------2")
            # print(self.weight_hh_l0[self.split["i"]])
            # h_t = h_t[0,:,:].t()
            # print(h_t)
            # print(self.weight_hh_l0[self.split["i"]] @ h_t)
            # t = s + self.weight_hh_l0[self.split["i"]] @ h_t
            # print(t)
            # print("---------------------------------3")
            # print(self.bias_hh_l0[self.split["i"]].unsqueeze(0).t())
            # i_t = torch.sigmoid(t + self.bias_hh_l0[self.split["i"]].unsqueeze(0).t())
            # print("\n")