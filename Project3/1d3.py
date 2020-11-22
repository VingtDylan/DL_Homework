import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.autograd import Variable

from lstm import *
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

def split_data_label(sequence, data_label):
    X = []
    Y = []
    for i in range(data_label.shape[0] - sequence):
        X.append(np.array(data_label.iloc[i: i + sequence, 1:-1], dtype = np.float32))
        Y.append(np.array(data_label.iloc[i + sequence, -1], dtype = np.float32))
    return X, Y

def main():
    """
    epochs = 150
    layers = 2
    input_size = 36
    hidden_size = 128
    lr = 0.001
    sequence_length = 7
    batch_size  = 16
    """
    args.epochs = 150 # 250 hhh
    args.layers = 2
    args.input_size = 36
    args.hidden_size = 128
    args.lr = 0.001
    args.sequence_length = 7
    args.batch_size  = 16

    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    copper_train_data_label_1d, copper_test_data_label_1d = load()
    # print(copper_train_data_label_1d)
    # print(copper_test_data_label_1d)
    
    sequence = args.sequence_length
    trainx, trainy = split_data_label(sequence = sequence,  data_label = copper_train_data_label_1d)
    testx, testy = split_data_label(sequence = sequence,  data_label = copper_test_data_label_1d)
    train_loader = DataLoader(dataset = Mydataset(trainx, trainy), batch_size = args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset = Mydataset(testx, testy), batch_size = args.batch_size, shuffle = False)

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size = 2, batch_first=args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) 

    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                # print(data,label)
                # print(data)
                data = data.squeeze(1).cuda()
                # print(data)
                pred = model(Variable(data).cuda())
                # print(pred.shape)
                # print(pred)
                # pred = pred[1,:,:] # lstm - 1
                # print(pred)
                # print(label)
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
        print(i, total_loss)
    
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
    if(acc > 55.01):
        torch.save(model, "models/1d/lstm_" + str(round(acc, 2)) + ".pkl")

if __name__ == "__main__":
    # TODO 38 features + more 
    main()