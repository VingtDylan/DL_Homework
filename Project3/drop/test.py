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
        Y.append(np.array(data_label.iloc[i + sequence, 37 :], dtype = np.float32))
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
    args.epochs = 10 # 250 55.88 decay 2887
    args.layers = 2
    args.input_size = 36
    args.hidden_size = 128
    args.lr = 0.001
    args.sequence_length = 14
    args.batch_size  = 16
    # 加载数据集并切分
    train_data_label_1d, test_data_label_1d = load()
    # print(train_data_label_1d.shape)
    # print(train_data_label_1d)
    # print(test_data_label_1d.shape)
    # print(test_data_label_1d)

    sequence = args.sequence_length
    trainx, trainy = split_data_label_2(sequence = sequence,  data_label = train_data_label_1d)
    testx, testy = split_data_label_2(sequence = sequence,  data_label = test_data_label_1d)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)
    # 定义模型并训练，测试
    model = Stock(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 6, batch_first=args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-7) 

    for i in range(args.epochs):
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = model(Variable(data).cuda())
                label = label.cuda()
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(i, loss.item())
    
    s = 0
    t = 0
    for idx, (data, label) in enumerate(train_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda()).cpu()
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            predicted = torch.where(pred > 0.5, one, zero)
            # print(predicted)
            # print(label)
            m, _ = predicted.shape
            ss = 0
            for i in range(m):
                ss += torch.equal(predicted[i],label[i])
            t += label.size(0)
            s += ss
    print(s,t)
    print(s * 100.0 / t)

    s = 0
    t = 0
    for idx, (data, label) in enumerate(test_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda()).cpu()
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            print(pred)
            predicted = torch.where(pred > 0.5, one, zero)
            print(predicted)
            print(label)
            m, _ = predicted.shape
            ss = 0
            for i in range(m):
                ss += torch.equal(predicted[i],label[i])
            t += label.size(0)
            s += ss
        break
    print(s,t)
    print(s * 100.0 / t)


if __name__ == "__main__":
    main()
    # a = Tensor([[1,2,3],[2,3,4]])
    # b = Tensor([[1,2,3],[3,2,5]])
    # m, n = a.shape
    # t = 0
    # for i in range(m):
    #     t += torch.equal(a[i],b[i])
    # print(t)