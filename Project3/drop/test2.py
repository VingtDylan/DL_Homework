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

    # #  more featues
    # # COMEX_train, COMEX_validation = load_COMEX_Train_Validation()
    # # for data in COMEX_train.values():
    # #     train_data_label = pd.merge(train_data_label,data, how = 'inner', on = 'date')
    # # for data in COMEX_validation.values():
    # #     validation_data_label = pd.merge(validation_data_label,data, how = 'inner', on = 'date')
    
    # 训练集 测试集的labels_1d
    LME_Label_train_1d = load_LME_Label_1d()
    LME_Label_Validation = load_Validation_Label()
    keys.append("Copper")
    for key in keys:
        LME_train_material_Label_1d = LME_Label_train_1d[key]
        LME_validation_material_1d = LME_Label_Validation[key + "1d"]
        LME_train_all = pd.merge(LME_train_all,LME_train_material_Label_1d, how = 'inner', on = 'date')
        LME_validation_all =  pd.merge(LME_validation_all,LME_validation_material_1d, how = 'inner', on = 'date')

    return LME_train_all, LME_validation_all

def split_data_label_2(sequence, data_label):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence):
        X.append(np.array(data_label.iloc[i: i + sequence, 1 : 37], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence, 37], dtype = np.float32))
    return X, Y

def main():
    # 固定随机种子
    set_seed(10)
    # 参数设置
    args.epochs = 300 # 250 55.88 decay 2887
    args.layers = 2
    args.input_size = 36
    args.hidden_size = 128
    args.lr = 0.0005
    args.sequence_length = 7
    args.batch_size  = 16
    # 加载数据集并切分
    train_data_label_1d, test_data_label_1d = load()
    # print(train_data_label_1d)
    sequence = args.sequence_length
    trainx, trainy = split_data_label_2(sequence = sequence,  data_label = train_data_label_1d)
    testx, testy = split_data_label_2(sequence = sequence,  data_label = test_data_label_1d)
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
    
    file_name = os.path.basename(__file__).split(".")[0]
    model.test_test(args, test_loader, file_name)

if __name__ == "__main__":
    main()