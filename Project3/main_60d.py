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

def load():
    # 训练集 测试集的features
    LME_train, LME_validation = load_LME_Train_Validation()
    LME_3M_train, LME_3M_validation = load_LME_3M_Train_Validation()
    # merge main features
    keys = ["Aluminium", "Lead", "Nickel", "Tin", "Zinc"]
    LME_train_all, LME_validation_all = LME_train["Copper"], LME_validation["Copper"]
    LME_3M_train_all, LME_3M_validation_all = LME_3M_train["Copper"], LME_3M_validation["Copper"]
    LME_train_all = pd.merge(LME_train_all,LME_3M_train_all, how = 'inner', on = 'date')
    LME_validation_all =  pd.merge(LME_validation_all,LME_3M_validation_all, how = 'inner', on = 'date')
    
    for key in keys:
        #  features cut
        LME_train_material, LME_validation_material = LME_train[key], LME_validation[key]
        LME_3M_train_material, LME_3M_validation_material = LME_3M_train[key], LME_3M_validation[key]
        # features merge
        LME_train_all = pd.merge(LME_train_all,LME_train_material, how = 'inner', on = 'date')
        LME_train_all = pd.merge(LME_train_all,LME_3M_train_material, how = 'inner', on = 'date')
        LME_validation_all =  pd.merge(LME_validation_all,LME_validation_material, how = 'inner', on = 'date')
        LME_validation_all =  pd.merge(LME_validation_all,LME_3M_validation_material, how = 'inner', on = 'date')
    
    # print(LME_train_all.columns.tolist()) 

    # COMEX features
    COMEX_train, COMEX_validation = load_COMEX_Train_Validation()
    for data in COMEX_train.values():
        LME_train_all = pd.merge(LME_train_all,data, how = 'inner', on = 'date')
    for data in COMEX_validation.values():
        LME_validation_all = pd.merge(LME_validation_all,data, how = 'inner', on = 'date')
    
    # Indices features
    Indices_train, Indices_validation = load_Indices_Train_Validation()
    for data in Indices_train.values():
        LME_train_all = pd.merge(LME_train_all,data, how = 'inner', on = 'date')
    for data in Indices_validation.values():
        LME_validation_all = pd.merge(LME_validation_all,data, how = 'inner', on = 'date')
    
    # 训练集 测试集的labels_60d
    LME_Label_train_60d = load_LME_Label_60d()
    LME_Label_Validation = load_Validation_Label()
    keys = ["Copper", "Aluminium", "Lead", "Nickel", "Tin", "Zinc"]
    for key in keys:
        LME_train_material_Label_60d = LME_Label_train_60d[key]
        LME_validation_material_60d = LME_Label_Validation[key + "60d"]
        LME_train_all = pd.merge(LME_train_all,LME_train_material_Label_60d, how = 'inner', on = 'date')
        LME_validation_all =  pd.merge(LME_validation_all,LME_validation_material_60d, how = 'inner', on = 'date')

    return LME_train_all, LME_validation_all

def main():
    # 固定随机种子
    set_seed(10)
    # 参数设置
    args.epochs = 150
    args.layers = 1
    args.input_size = 72
    args.hidden_size = 132 
    args.lr = 0.0001  
    args.sequence_length = 14
    args.batch_size  = 8
    """ 150 1 72 132 0.0001 14 8
    ------------训练集--------------
    金属次序: Copper Aluminium, Lead, Nickel, Tin, Zinc
    训练集样本个数:
    801 801 801 801 801 801
    训练集正确预测个数:
    721 621 720 586 478 709
    训练集准确率:
    90.01% 77.53% 89.89% 73.16% 59.68% 88.51%
    平均准确率:
    79.8
    ------------测试集--------------
    金属次序: Copper Aluminium, Lead, Nickel, Tin, Zinc
    测试集样本个数:
    142 142 142 142 142 142
    测试集正确预测个数:
    74 87 49 48 54 71
    测试集准确率:
    52.11 61.27 34.51 33.8 38.03 50.0
    平均准确率:
    44.95
    base: 63.97, baseline: 77.01
    """
    # 加载数据集并切分
    train_data_label_60d, test_data_label_60d = load()

    sequence = args.sequence_length
    trainx, trainy = split_data_label_merge(sequence = sequence,  data_label = train_data_label_60d, delay = 60)
    testx, testy = split_data_label_merge(sequence = sequence,  data_label = test_data_label_60d, delay = 60)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)
    # 定义模型并训练，测试
    model = LSTM_Stock(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 2, batch_first=args.batch_first)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-7) 

    model.train(args, train_loader, criterion, optimizer)
    model.train_test(args, train_loader)

    file_name = "60d"
    model.test_test(args, test_loader, file_name)
if __name__ == "__main__":
    main()

