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

def load():
    COMEX_Copper_train_names = ["id","date","Coppper_Open","Coppper_High","Coppper_Low","Coppper_Close","Coppper_Volumn","Coppper_Open_Interest"]
    COMEX_Copper_train_path = "Train_data" + "/" + "COMEX" + "/"+"COMEX_Copper_train" + ".csv"
    COMEX_Copper_train = pd.read_csv(COMEX_Copper_train_path, skiprows = 1,names = COMEX_Copper_train_names)
    COMEX_Copper_train.drop(labels = "id", axis = 1,inplace = True)

    # AI_train_names = ["id","date","attr","a","b","c","d"]
    # Al_train_path = "Train_data" + "/" + "LME" + "/"+"LMEAluminium3M_train" + ".csv"
    AI_train_names = ["id","date","attr"]
    Al_train_path = "Train_data" + "/" + "LME" + "/"+"LMEAluminium_OI_train" + ".csv"
    Al_train = pd.read_csv(Al_train_path, skiprows = 1,names = AI_train_names)
    Al_train.drop(labels = "id", axis = 1,inplace = True)

    Al_label_1d_name = ["id","date", "label"]
    Al_label_1d_path = "Train_data" + "/" + "Label" + "/"+"Label_LMEAluminium_train_1d" + ".csv"
    Al_label_1d = pd.read_csv(Al_label_1d_path,skiprows = 1, names = Al_label_1d_name)
    Al_label_1d.drop(labels = "id", axis = 1,inplace = True)


    Al_train_label_1d = pd.merge(Al_train,COMEX_Copper_train, how = 'inner', on = 'date')
    Al_train_label_1d = pd.merge(Al_train_label_1d,Al_label_1d, how = 'inner', on = 'date')
    Al_train_label_1d.dropna(axis = 0, how = "any",inplace = True)


    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    Attr = Al_train_label_1d.columns.tolist()[1:]
    for att in Attr:
        Al_train_label_1d[att] = Al_train_label_1d[[att]].apply(max_min_scaler)
    Al_train_label_1d.to_csv("Al_train_label_1d.csv",index=False,sep=',')
    
    # s = Al_train_label_1d.isnull().any(axis=0)
    # print(s)
    return Al_train_label_1d

class Mydataset(Dataset):
    def __init__(self, xx, yy):
        self.x = xx
        self.y = yy

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        return x1, y1

    def __len__(self):
        return len(self.x)

def main():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    al_train_label_1d = load()
    print(al_train_label_1d)
    pass
    
    sequence = args.sequence_length
    X = []
    Y = []
    for i in range(al_train_label_1d.shape[0] - sequence):
        X.append(np.array(al_train_label_1d.iloc[i: i + sequence, 1:-1], dtype = np.float32))
        Y.append(np.array(al_train_label_1d.iloc[i + sequence, -1], dtype = np.float32))

    total_len = len(Y)
    args.input_size = X[0].shape[1]

    trainx, trainy = X[:int(0.9 * total_len)], Y[:int(0.9 * total_len)]
    testx, testy = X[int(0.9 * total_len):], Y[int(0.9 * total_len):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), batch_size=args.batch_size,shuffle = False)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=args.batch_size, shuffle = False)


    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, batch_first=args.batch_first)
    model.to(args.device)
    print(model)

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) 

    s = 1
    for i in range(args.epochs):
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
            s = s + 1
        print(i, loss.item())
        # for name,param in model.named_parameters():
        #     print(name,param)
        # print("\n")
        #     # if s > 100:
        #     break
        # break
    
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
            print(predicted)
            print(label)
    print(s,t)
    print(s * 100.0 / t)
# 47.4320
# 48.3384
# 
if __name__ == "__main__":
    # 6features的垃圾
    main()