import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable

from lstm2 import *
from parse import *
from MyDataLoader import *

def load():
    # 训练集 测试集的features
    LME_train, LME_validation = load_LME_Train_Validation()
    LME_3M_train, LME_3M_validation = load_LME_3M_Train_Validation()
    LME_train_Copper, LME_validation_Copper = LME_train["Copper"], LME_validation["Copper"]
    LME_3M_train_Copper, LME_3M_validation_Copper = LME_3M_train["Copper"], LME_3M_validation["Copper"]
    # features merge
    train_data_label = pd.merge(LME_train_Copper,LME_3M_train_Copper, how = 'inner', on = 'date')
    validation_data_label =  pd.merge(LME_validation_Copper,LME_3M_validation_Copper, how = 'inner', on = 'date')
    #  more featues - COMEX
    COMEX_train, COMEX_validation = load_COMEX_Train_Validation()
    for data in COMEX_train.values():
        train_data_label = pd.merge(train_data_label,data, how = 'inner', on = 'date')
    for data in COMEX_validation.values():
        validation_data_label = pd.merge(validation_data_label,data, how = 'inner', on = 'date')
    # more features - Indice
    Indices_train, Indices_validation =  load_Indices_Train_Validation()
    for data in Indices_train.values():
        train_data_label = pd.merge(train_data_label,data, how = 'inner', on = 'date')
    for data in Indices_validation.values():
        validation_data_label = pd.merge(validation_data_label,data, how = 'inner', on = 'date')
    # 训练集 测试集的labels_20d
    LME_Label_train_20d = load_LME_Label_20d()
    LME_Label_Validation = load_Validation_Label()
    LME_train_Copper_Label_20d = LME_Label_train_20d["Copper"]
    LME_validation_Copper_20d = LME_Label_Validation["Copper" + "20d"]
    # merge data and label
    train_data_label = pd.merge(train_data_label,LME_train_Copper_Label_20d, how = 'inner', on = 'date')
    validation_data_label =  pd.merge(validation_data_label,LME_validation_Copper_20d, how = 'inner', on = 'date')

    return train_data_label, validation_data_label

def split_data_label(sequence, data_label):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence - 20):
        X.append(np.array(data_label.iloc[i: i + sequence, 1:-1], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence + 20, -1], dtype = np.float32))
    return X, Y

def main():
    """
    epochs = 300 # 350 
    layers = 3
    input_size = 42
    hidden_size = 128
    lr = 0.001
    sequence_length = 5
    batch_size  = 16
    """
    args.epochs = 300 # 330 64.8936 7273
    args.layers = 3
    args.input_size = 42 # 36
    args.hidden_size = 128
    args.lr = 0.0005
    args.sequence_length = 7
    args.batch_size  = 16
    # args.dropout = 0.1

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    copper_train_data_label_20d, copper_test_data_label_20d = load()
    # print(copper_train_data_label_20d) 
    # print(copper_test_data_label_20d) 
    
    sequence = args.sequence_length
    trainx, trainy = split_data_label(sequence = sequence,  data_label = copper_train_data_label_20d)
    testx, testy = split_data_label(sequence = sequence,  data_label = copper_test_data_label_20d)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)

    model = lstm2(input_size = args.input_size, hidden_size = args.hidden_size, num_layers = args.layers , output_size = 2, batch_first = args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-7) 

    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data = data.squeeze(1).cuda()
                pred = model(Variable(data).cuda())
                label = label.cuda()
                # label = label.unsqueeze(1).cuda()
                # print(label)
                # print(label.shape)
            # else:
            #     data1 = data.squeeze(1)
            #     pred = model(Variable(data1))
            #     pred = pred[1, :, :]
            #     label = label.unsqueeze(1)
            # print(pred,label)
            loss = criterion(pred, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(i, loss.item(), total_loss)

    s = 0
    t = 0
    for idx, (data, label) in enumerate(train_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda())
            _, predicted = torch.max(pred.data, 1)
            t += label.size(0)
            s += (predicted.cpu() == label.cpu()).sum()
    print(s,t)
    print(s * 100.0 / t)

    s = 0
    t = 0
    for idx, (data, label) in enumerate(test_loader):
        if args.useGPU:
            data = data.squeeze(1).cuda()
            pred = model(Variable(data).cuda())
            _, predicted = torch.max(pred.data, 1)
            t += label.size(0)
            s += (predicted.cpu() == label.cpu()).sum()
            # print(pred)
            # print(predicted)
            # print(label)
    print(s,t)
    print(s * 100.0 / t)

    acc = s * 100.0 / t
    acc = acc.item()
    print("base : 63.57; baseline: 70.29")
    if(acc > 70.29):
        torch.save(model, "models/20d/lstm_" + str(round(acc, 2)) + ".pkl")
        print("model with acc " + str(round(acc, 2)) + "%" + " is saved!")

if __name__ == "__main__":
    # 38 features 
    main()