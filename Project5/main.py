
import os
from os import path

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import NLLLoss, CrossEntropyLoss, BCEWithLogitsLoss 
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from sklearn.metrics import roc_auc_score  

from matplotlib import pyplot as plt

from model.NRMS import NRMS
from lib.dataset import MyDataSet
from lib.utils import set_seed
from lib.config import nrms
from final_data_preprocess_drop import generate_word_embedding

def main():
    # torch.autograd.set_detect_anomaly(True)
    # 固定随机种子
    set_seed(10)
    device = torch.device(f"cuda:{nrms.device}" if torch.cuda.is_available() else "cpu")
    # prepare pretrained embeddings
    print(f"Load pretrained embeddings")
    glove_embedding_matrix_path = './Processed/large_glove_embedding_matrix_' + str(nrms.word_embedding_dim) + 'd.txt'
    pretrained_entity_embedding = generate_word_embedding(glove_embedding_matrix_path, flag = False)
    pretrained_entity_embedding = pretrained_entity_embedding.to(device)
    # model
    print(f"Create model")
    model = NRMS(nrms, pretrained_entity_embedding).to(device)
    print(model)
    # parpare data
    print(f"Load train data")
    train_clicked_news = \
        pickle.load(open("./Processed/large_clicked_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    train_candidate_news = \
        pickle.load(open("./Processed/large_candidate_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    train_label = \
        pickle.load(open("./Processed/large_label_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load dev data")
    dev_clicked_news = \
        pickle.load(open("./Processed/large_clicked_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_candidate_news = \
        pickle.load(open("./Processed/large_candidate_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_label = \
        pickle.load(open("./Processed/large_label_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load test data")
    test_clicked_news = \
        pickle.load(open("./Processed/large_clicked_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    test_candidate_news = \
        pickle.load(open("./Processed/large_candidate_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    test_label = \
        pickle.load(open("./Processed/large_label_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load dev_test data")
    dev_test_clicked_news = \
        pickle.load(open("./Processed/large_clicked_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_candidate_news = \
        pickle.load(open("./Processed/large_candidate_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_label = \
        pickle.load(open("./Processed/large_label_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    print("train data counters: {}".format(len(train_label)))
    print("dev data counters: {}".format(len(dev_label)))
    print("test data counters: {}".format(len(test_label)))
    print("dev_test data counters: {}".format(len(dev_test_label)))

    train_dataset = MyDataSet(train_clicked_news, train_candidate_news, train_label)
    dev_dataset = MyDataSet(dev_clicked_news, dev_candidate_news, dev_label)
    test_dataset = MyDataSet(test_clicked_news, test_candidate_news, test_label)
    dev_test_dataset = MyDataSet(dev_test_clicked_news, dev_test_candidate_news, dev_test_label)
    train_dataloader = DataLoader(train_dataset, batch_size = nrms.batch_size, shuffle = True)
    dev_dataloader = DataLoader(dev_dataset, batch_size = nrms.batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    dev_test_dataloader = DataLoader(dev_test_dataset, batch_size = 1, shuffle = False)

    # 生成验证集的真实标签表，生成后即可注释掉
    print("---------------------------Generate my dev truth--Process--------------------------------")
    with open("dev_truth.txt", 'w') as f:
        for _idx, (_, __, _label) in enumerate(dev_test_dataloader):    
            __label = [label.detach().item() for label in _label]
            f.write(str(_idx + 1))
            f.write(" [")
            sortStr = [str(label) for label in __label]
            f.write(",".join(sortStr))
            f.write("]")
            f.write("\n")
        print("\033[0;35;40m\tdev truth table out!\033[0m")
        
    print("------------------------------Train--Process--------------------------------")
    optimizer = Adam(model.parameters(), lr = nrms.learning_rate)
    # optimizer = Adam(model.parameters(), lr = nrms.learning_rate, weight_decay = 5e-4)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = CrossEntropyLoss().to(device)
    LR = []
    # criterion = BCEWithLogitsLoss().to(device) 
    # criterion = NLLLoss().to(device)
    for epoch in range(1, 1 + nrms.num_epochs):
        model.train()
        total_loss = 0
        total_counter = 0
        for _idx, (_clicked_news, _candidate_news, _label) in enumerate(train_dataloader):
            _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.to(device)
            _labels = torch.Tensor(np.argmax(_label.numpy(), axis = 1)).long().to(device)
            _pred = model(_clicked_news, _candidate_news)
            _loss = criterion(_pred, _labels)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            total_loss += _loss.item()
            total_counter += _pred.shape[0]
            if _idx % 500 == 0:
                print("  Epoch {} batch {} loss: {}".format(epoch, _idx, _loss.item()))
        ExpLR.step()
        lr = ExpLR.get_last_lr()[0]
        LR.append(lr)
        print("\033[0;35;40m\t--------------------------Epoch:" + str(epoch) + "------------------------")
        print(LR)
        print("\033[0;35;40m\tEpoch {} train average loss: {}\033[0m".format(epoch, total_loss / total_counter))
        
        model.eval()
        total_loss = 0
        total_counter = 0
        total_correct = 0
        Auc = []
        for _idx, (_clicked_news, _candidate_news, _label) in enumerate(dev_dataloader):
            _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.to(device)
            _labels = torch.Tensor(np.argmax(_label.numpy(), axis=1)).long().to(device)
            _pred = model(_clicked_news, _candidate_news)
            _loss = criterion(_pred, _labels)
            correct = sum(np.argmax(_pred.cpu().detach().numpy(), axis=1) == np.argmax(_label.cpu().detach().numpy(), axis=1))
            total_loss += _loss.item()
            total_counter += _pred.shape[0]
            total_correct += correct
            __pred = _pred.cpu().detach().numpy()
            __label = _label.cpu().detach().numpy()
            for idx, s_pred in enumerate(__pred):
                Auc.append(roc_auc_score(__label[idx], s_pred))
        print("\033[0;35;40m\tEpoch {} dev average loss: {} acc {:.2%} auc {:.2}\033[0m".format(epoch, total_loss / total_counter, total_correct / total_counter, sum(Auc) / len(Auc)))
        torch.save(model.state_dict(), "large_model-{}-{:.3f}-{:.3f}-{:.2f}.pkl".format(epoch, total_loss / total_counter, total_correct / total_counter, sum(Auc) / len(Auc)))

    print("-------------------------Learning rate in training process-----------------------")
    print(LR)
    EPOCHS = np.arange(1, 1 + nrms.num_epochs)
    plt.figure()
    plt.plot(EPOCHS, LR)
    plt.xlim(1, 1 + nrms.num_epochs)
    plt.xticks(EPOCHS)
    plt.show()

if __name__ == "__main__":
    main()
    '''weight_decay + lr
    
    '''