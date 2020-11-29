import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable

from Parse import *
from Util import *
from MyLSTM_Stock import *
from MyDataLoader import *

def load(args):
    # 训练集 测试集的features
    LME_train, LME_validation = load_LME_Train_Validation(args.sequence_length + args.delay)
    LME_3M_train, LME_3M_validation = load_LME_3M_Train_Validation(args.sequence_length + args.delay)
    # merge main features
    keys = ["Aluminium", "Lead", "Nickel", "Tin", "Zinc"]
    LME_train_all, LME_validation_all = LME_train["Copper"], LME_validation["Copper"]
    LME_3M_train_all, LME_3M_validation_all = LME_3M_train["Copper"], LME_3M_validation["Copper"]
    LME_train_all = pd.merge(LME_train_all,LME_3M_train_all, how = 'outer', on = 'date', sort = True)
    LME_train_all.fillna(method="ffill", inplace=True)
    LME_train_all.fillna(method="bfill", inplace=True)
    LME_validation_all =  pd.merge(LME_validation_all,LME_3M_validation_all, how = 'outer', on = 'date', sort = True)
    LME_validation_all.fillna(method="ffill", inplace=True)
    LME_validation_all.fillna(method="bfill", inplace=True)
    
    for key in keys:
        #  features cut
        LME_train_material, LME_validation_material = LME_train[key], LME_validation[key]
        LME_3M_train_material, LME_3M_validation_material = LME_3M_train[key], LME_3M_validation[key]
        # features merge
        LME_train_all = pd.merge(LME_train_all,LME_train_material, how = 'outer', on = 'date', sort = True)
        LME_train_all.fillna(method="ffill", inplace=True)
        LME_train_all.fillna(method="bfill", inplace=True)
        LME_train_all = pd.merge(LME_train_all,LME_3M_train_material, how = 'outer', on = 'date', sort = True)
        LME_train_all.fillna(method="ffill", inplace=True)
        LME_train_all.fillna(method="bfill", inplace=True)
        LME_validation_all =  pd.merge(LME_validation_all,LME_validation_material, how = 'outer', on = 'date', sort = True)
        LME_validation_all.fillna(method="ffill", inplace=True)
        LME_validation_all.fillna(method="bfill", inplace=True)
        LME_validation_all =  pd.merge(LME_validation_all,LME_3M_validation_material, how = 'outer', on = 'date', sort = True)
        LME_validation_all.fillna(method="ffill", inplace=True)
        LME_validation_all.fillna(method="bfill", inplace=True)

    # Indices features
    # Indices_train, Indices_validation = load_Indices_Train_Validation(delay = args.sequence_length)
    # for data in Indices_train.values():
    #     LME_train_all = pd.merge(LME_train_all,data, how = 'outer', on = 'date', sort = True)
    #     LME_train_all.fillna(method="ffill", inplace=True)
    #     LME_train_all.fillna(method="bfill", inplace=True)
    # for data in Indices_validation.values():
    #     LME_validation_all = pd.merge(LME_validation_all,data, how = 'outer', on = 'date', sort = True)
    #     LME_validation_all.fillna(method="ffill", inplace=True)
    #     LME_validation_all.fillna(method="bfill", inplace=True)

    # COMEX features
    COMEX_train, COMEX_validation = load_COMEX_Train_Validation(delay = args.sequence_length + args.delay)
    for data in COMEX_train.values():
        LME_train_all = pd.merge(LME_train_all,data, how = 'outer', on = 'date', sort = True)
        LME_train_all.fillna(method="ffill", inplace=True)
        LME_train_all.fillna(method="bfill", inplace=True)
    for data in COMEX_validation.values():
        LME_validation_all = pd.merge(LME_validation_all,data, how = 'outer', on = 'date', sort = True)
        LME_validation_all.fillna(method="ffill", inplace=True)
        LME_validation_all.fillna(method="bfill", inplace=True)
    
    # 训练集 测试集的labels_20d
    LME_Label_train_20d, LME_Label_train_20d_extra = load_LME_Label_20d(delay = args.sequence_length + args.delay)
    LME_Label_Validation = load_Validation_Label()
    keys = ["Copper", "Aluminium", "Lead", "Nickel", "Tin", "Zinc"]
    for key in keys:
        LME_train_material_Label_20d = LME_Label_train_20d[key]
        LME_validation_material_20d = LME_Label_Validation[key + "20d"]
        LME_validation_material_20d = pd.concat([LME_Label_train_20d_extra[key], LME_validation_material_20d], axis = 0)
        LME_train_all = pd.merge(LME_train_all,LME_train_material_Label_20d, how = 'outer', on = 'date', sort = True)
        LME_train_all.fillna(method="ffill", inplace=True)
        LME_train_all.fillna(method="bfill", inplace=True)
        LME_validation_all =  pd.merge(LME_validation_all,LME_validation_material_20d, how = 'outer', on = 'date', sort = True)
        LME_validation_all.fillna(method="ffill", inplace=True)
        LME_validation_all.fillna(method="bfill", inplace=True)

    return LME_train_all, LME_validation_all

def main():
    # 固定随机种子
    set_seed(10)
    # 参数设置
    args.epochs = 130  
    args.layers = 2
    args.input_size = 66
    args.hidden_size = 256
    args.lr = 0.001
    args.sequence_length = 21
    args.batch_size  = 32
    args.delay = 20
    # 加载数据集并切分
    train_data_label_20d, test_data_label_20d = load(args)
    sequence = args.sequence_length
    trainx, trainy, valx, valy = split_data_label_merge(sequence = sequence,  data_label = train_data_label_20d, delay = 20, split = True)
    testx, testy = split_data_label_merge(sequence = sequence,  data_label = test_data_label_20d, delay = 20, split = False)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    val_loader = DataLoader(dataset = Mydataset(valx, valy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)
    # 定义模型并训练，测试
    model = LSTM_Stock(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 2, batch_first=args.batch_first)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-5) 

    model.train(args, train_loader, val_loader, criterion, optimizer)
    model.train_test(args, train_loader)

    file_name = "20d"
    model.test_test(args, test_loader, file_name)

if __name__ == "__main__":
    main()

