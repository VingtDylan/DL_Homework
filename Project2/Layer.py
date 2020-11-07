import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.functional import fold, unfold
from typing import Union, Tuple, List

class Layer(nn.Module):
    def __init__(self):
        super(Layer,self).__init__()
        self.cache = None
    
    def _display(self):
        print(self.__class__.__name__)

class MyLinear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor)->Tensor:
        self.cache = input.matmul(self.weight.t())
        if self.bias is not None:
            self.cache += self.bias
        return self.cache

class MyTanh(Layer):
    def __init__(self):
        super(MyTanh,self).__init__()

    def forward(self, input: Tensor)->Tensor:
        self.cache = torch.tanh(input)
        return self.cache

class MyConv2d(Layer):
    T = int
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0, bias: bool = True):
        super(MyConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if not isinstance(self.kernel_size, tuple):
            self.kernel_size = (self.kernel_size,self.kernel_size)
        self.stride = stride
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride,self.stride)
        self.padding = padding

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        batch_size, _, height, width = input.shape
        out_height = height - self.kernel_size[0] + 1
        out_width = width - self.kernel_size[1] + 1
        self.cache = torch.zeros((batch_size, self.out_channels, out_height, out_width))
        if input.is_cuda:
            self.cache = self.cache.cuda()
        # brute-force
        # for n in range(batch_size):
        #     for c in range(self.out_channels):
        #         for h in range(out_height):
        #             for w in range(out_width):
        #                 self.cache[n,c,h,w] = torch.sum(self.weight[c,:,:,:] * input[n,:,h : h + self.kernel_size[0], w : w + self.kernel_size[1]]) + self.bias[c]

        _input = unfold(input,self.kernel_size)
        _e = _input.transpose(1,2).matmul(self.weight.view(self.weight.size(0),-1).t()) + self.bias
        _e = _e.transpose(1,2)
        self.cache = fold(_e,(out_height,out_width),(1,1))
        return self.cache

class MyAvgPool2d(Layer):
    T = int
    def __init__(self, kernel_size: Union[T, Tuple[T, T]], stride: Union[T, Tuple[T, T]] = None, padding: Union[T, Tuple[T, T]] = 0) -> None:
        super(MyAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        if not isinstance(self.kernel_size, tuple):
            self.kernel_size = (self.kernel_size,self.kernel_size)
        self.stride = stride if (stride is not None) else self.kernel_size
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        batch_size, channels, height, width = input.shape
        out_height = int((height - self.kernel_size[0]) / self.stride[0] + 1)
        out_width = int((width - self.kernel_size[1]) / self.stride[1] + 1)
        self.cache = torch.zeros((batch_size, channels, out_height, out_width))
        if input.is_cuda:
            self.cache = self.cache.cuda()
        # brute-force
        # for i in range(out_height):
        #    for j  in range(out_width):
        #        _mask = input[:,:,i * self.stride[0]: i * self.stride[0] + self.kernel_size[0],j * self.stride[1]: j * self.stride[1] + self.kernel_size[1]]
        #        self.cache[:,:,i,j] = torch.mean(_mask,dim = (2,3))
        input = input.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        self.cache = input.contiguous().view(input.size()[:4] + (-1,)).mean(dim = -1)
        # print(self.cache.shape)
        return self.cache