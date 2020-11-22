import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable

from Stock import *
from parse import *
from MyDataLoader import *
from Util import *

def load():
    # 训练集 测试集的features
    LME_train, LME_validation = load_LME_Train_Validation()
    LME_3M_train, LME_3M_validation = load_LME_3M_Train_Validation()
    LME_train_Copper, LME_validation_Copper = LME_train["Copper"], LME_validation["Copper"]
    LME_3M_train_Copper, LME_3M_validation_Copper = LME_3M_train["Copper"], LME_3M_validation["Copper"]
    # features merge
    train_data_label = pd.merge(LME_train_Copper,LME_3M_train_Copper, how = 'inner', on = 'date')
    validation_data_label =  pd.merge(LME_validation_Copper,LME_3M_validation_Copper, how = 'inner', on = 'date')
    #  more featues
    COMEX_train, COMEX_validation = load_COMEX_Train_Validation()
    for data in COMEX_train.values():
        train_data_label = pd.merge(train_data_label,data, how = 'inner', on = 'date')
    for data in COMEX_validation.values():
        validation_data_label = pd.merge(validation_data_label,data, how = 'inner', on = 'date')
    # 训练集 测试集的labels_1d
    LME_Label_train_1d = load_LME_Label_1d()
    LME_Label_Validation = load_Validation_Label()
    LME_train_Copper_Label_1d = LME_Label_train_1d["Copper"]
    LME_validation_Copper_1d = LME_Label_Validation["Copper" + "1d"]
    # merge data and label
    train_data_label = pd.merge(train_data_label,LME_train_Copper_Label_1d, how = 'inner', on = 'date')
    validation_data_label =  pd.merge(validation_data_label,LME_validation_Copper_1d, how = 'inner', on = 'date')

    return train_data_label, validation_data_label

def main():
    # 固定随机种子
    set_seed(10)
    # 参数设置
    args.epochs = 250 # 250 55.88 decay 2887
    args.layers = 2
    args.input_size = 36
    args.hidden_size = 128
    args.lr = 0.0005
    args.sequence_length = 7
    args.batch_size  = 16
    # 加载数据集并切分
    copper_train_data_label_1d, copper_test_data_label_1d = load()
    sequence = args.sequence_length
    trainx, trainy = split_data_label(sequence = sequence,  data_label = copper_train_data_label_1d)
    testx, testy = split_data_label(sequence = sequence,  data_label = copper_test_data_label_1d)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)
    # 定义模型并训练，测试
    model = Stock(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 2, batch_first=args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-7) 

    model.train(args, train_loader, criterion, optimizer)
    model.train_test(args, train_loader)
    model.test_test(args, test_loader)

if __name__ == "__main__":
    main()