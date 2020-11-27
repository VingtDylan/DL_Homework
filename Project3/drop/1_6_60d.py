
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable

from Stock2 import *
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

    # COMEX features
    COMEX_train, COMEX_validation = load_COMEX_Train_Validation()
    for data in COMEX_train.values():
        LME_train_all = pd.merge(LME_train_all,data, how = 'inner', on = 'date')
    for data in COMEX_validation.values():
        LME_validation_all = pd.merge(LME_validation_all,data, how = 'inner', on = 'date')
    # indices features
    # Indices_train, Indices_validation = load_Indices_Train_Validation()
    # for data in Indices_train.values():
    #     LME_train_all = pd.merge(LME_train_all,data, how = 'inner', on = 'date')
    # for data in Indices_validation.values():
    #     LME_validation_all = pd.merge(LME_validation_all,data, how = 'inner', on = 'date')
    
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

def split_data_label_2(sequence, data_label):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence - 60):
        X.append(np.array(data_label.iloc[i: i + sequence, 1 : -6], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence + 60, -6 : ], dtype = np.float32))
    return X, Y

def main():
    # 固定随机种子
    # set_seed(10)
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # 参数设置
    args.epochs = 500
    args.layers = 2
    args.input_size = 66
    args.hidden_size = 256
    args.lr = 0.001
    args.sequence_length = 14
    args.batch_size  = 16
    """

    """
    # 加载数据集并切分
    train_data_label_60d, test_data_label_60d = load()
    # print(train_data_label_60d.shape)
    # print(train_data_label_60d)
    # print(test_data_label_60d.shape)
    # print(test_data_label_60d)

    sequence = args.sequence_length
    trainx, trainy = split_data_label_2(sequence = sequence,  data_label = train_data_label_60d)
    testx, testy = split_data_label_2(sequence = sequence,  data_label = test_data_label_60d)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)
    # 定义模型并训练，测试
    model = Stock2(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 2, batch_first=args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-7) 

    for epoch in range(args.epochs):
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = model(Variable(data).cuda())
                label = label.cuda()
            loss = criterion(pred[0], label[:,0].long())
            for i in range(1, 6):
                loss += criterion(pred[i], label[:,i].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
    
    s = [0] * 6
    t = [0] * 6 
    for idx, (data, label) in enumerate(train_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda())[0].cpu()
            _, predicted = torch.max(pred.data, 1)
            for i in range(6):
                t[i] += label.size(0)
                s[i] += (predicted.cpu() == label[:,i].cpu()).sum().item()
    acc = [round(s[i] * 100.0 / t[i], 2) for i in range(6)]
    print(s)
    print(t)
    print(acc)

    print("------------------------------------------")

    s = [0] * 6
    t = [0] * 6 
    for idx, (data, label) in enumerate(test_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda())[0].cpu()
            _, predicted = torch.max(pred.data, 1)
            for i in range(6):
                t[i] += label.size(0)
                s[i] += (predicted.cpu() == label[:,i].cpu()).sum().item()
    acc = [round(s[i] * 100.0 / t[i], 2) for i in range(6)]
    print(s)
    print(t)
    print(acc)

if __name__ == "__main__":
    main()
    # a = Tensor([[1,2,3],[2,3,4]])
    # b = Tensor([[1,2,3],[3,2,5]])
    # m, n = a.shape
    # t = 0
    # for i in range(m):
    #     t += torch.equal(a[i],b[i])
    # print(t)