
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

from model.NRMS import NRMS
from dataset import MyDataSet
from lib.utils import set_seed
from lib.config import nrms
from final_data_preprocess_drop import generate_word_embedding

from evaluate import *

def main():
    # torch.autograd.set_detect_anomaly(True)
    # 固定随机种子
    set_seed(10)
    device = torch.device(f"cuda:{nrms.device}" if torch.cuda.is_available() else "cpu")
    # prepare pretrained embeddings
    print(f"Load pretrained embeddings")
    glove_embedding_matrix_path = 'large_glove_embedding_matrix_' + str(nrms.word_embedding_dim) + 'd.txt'
    pretrained_entity_embedding = generate_word_embedding(glove_embedding_matrix_path, flag = False)
    pretrained_entity_embedding = pretrained_entity_embedding.to(device)
    # model
    print(f"Create model")
    model = NRMS(nrms, pretrained_entity_embedding).to(device)
    print(model)
    # parpare data
    print(f"Load train data")
    train_clicked_news = \
        pickle.load(open("large_clicked_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    train_candidate_news = \
        pickle.load(open("large_candidate_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    train_label = \
        pickle.load(open("large_label_train_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load dev data")
    dev_clicked_news = \
        pickle.load(open("large_clicked_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_candidate_news = \
        pickle.load(open("large_candidate_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_label = \
        pickle.load(open("large_label_dev_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load test data")
    test_clicked_news = \
        pickle.load(open("large_clicked_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    test_candidate_news = \
        pickle.load(open("large_candidate_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    test_label = \
        pickle.load(open("large_label_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))

    print(f"Load dev_test data")
    dev_test_clicked_news = \
        pickle.load(open("large_clicked_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_candidate_news = \
        pickle.load(open("large_candidate_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_label = \
        pickle.load(open("large_label_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    print("train data counters: {}".format(len(train_label)))
    print("dev data counters: {}".format(len(dev_label)))
    print("test data counters: {}".format(len(test_label)))
    print("dev_test data counters: {}".format(len(dev_test_label)))

    # test 237072
    # dev_test 376471

    # sample1 = 10 # 1435902
    # train_clicked_news = train_clicked_news[0:sample1]
    # train_candidate_news = train_candidate_news[0:sample1]
    # train_label = train_label[0:sample1]
    # sample2 = 10 # 666309
    # dev_clicked_news = dev_clicked_news[0:sample2]
    # dev_candidate_news = dev_candidate_news[0:sample2]
    # dev_label = dev_label[0:sample2]
    # sample3 = 10 # 666309
    # test_clicked_news = test_clicked_news[0:sample3]
    # test_candidate_news = test_candidate_news[0:sample3]
    # test_label = test_label[0:sample3]
    # sample4 = 10 # 666309
    # dev_test_clicked_news = dev_test_clicked_news[0:sample4]
    # dev_test_candidate_news = dev_test_candidate_news[0:sample4]
    # dev_test_label = dev_test_label[0:sample4]

    train_dataset = MyDataSet(train_clicked_news, train_candidate_news, train_label)
    dev_dataset = MyDataSet(dev_clicked_news, dev_candidate_news, dev_label)
    test_dataset = MyDataSet(test_clicked_news, test_candidate_news, test_label)
    dev_test_dataset = MyDataSet(dev_test_clicked_news, dev_test_candidate_news, dev_test_label)
    train_dataloader = DataLoader(train_dataset, batch_size = nrms.batch_size, shuffle = True)
    dev_dataloader = DataLoader(dev_dataset, batch_size = nrms.batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    dev_test_dataloader = DataLoader(dev_test_dataset, batch_size = 1, shuffle = False)

    # print("---------------------------Generate my dev truth--Process--------------------------------")
    # with open("dev_truth.txt", 'w') as f:
    #     for _idx, (_, __, _label) in enumerate(dev_test_dataloader):    
    #         __label = [label.detach().item() for label in _label]
    #         f.write(str(_idx + 1))
    #         f.write(" [")
    #         sortStr = [str(label) for label in __label]
    #         f.write(",".join(sortStr))
    #         f.write("]")
    #         f.write("\n")
    #     print("\033[0;35;40m\tdev truth table out!\033[0m")
        
    print("------------------------------Train--Process--------------------------------")
    optimizer = Adam(model.parameters(), lr = nrms.learning_rate, weight_decay = 5e-4)
    criterion = CrossEntropyLoss().to(device)
    # criterion = BCEWithLogitsLoss().to(device) 
    # criterion = NLLLoss().to(device)
    for epoch in range(1, 1 + nrms.num_epochs):
        model.train()
        total_loss = 0
        total_counter = 0
        for _idx, (_clicked_news, _candidate_news, _label) in enumerate(train_dataloader):
            _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.to(device)
            # _labels = _label.long().to(device)
            _labels = torch.Tensor(np.argmax(_label.numpy(), axis = 1)).long().to(device)
            _pred = model(_clicked_news, _candidate_news)
            # _pred = torch.log(_pred)
            # print(_pred[-1])
            # print(torch.softmax(_pred, dim = 1))
            # print(_pred)
            # print(_labels)
            _loss = criterion(_pred, _labels)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            total_loss += _loss.item()
            total_counter += _pred.shape[0]
            # break
            if _idx % 50 == 0:
                print("  Epoch {} batch {} loss: {}".format(epoch, _idx, _loss.item()))
        print("\033[0;35;40m\tEpoch {} train average loss: {}\033[0m".format(epoch, total_loss / total_counter))
        
        model.eval()
        total_loss = 0
        total_counter = 0
        total_correct = 0
        Auc = []
        for _idx, (_clicked_news, _candidate_news, _label) in enumerate(dev_dataloader):
            _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.to(device)
            # _labels = _label.long().to(device)
            _labels = torch.Tensor(np.argmax(_label.numpy(), axis=1)).long().to(device)
            _pred = model(_clicked_news, _candidate_news)
            # _pred = torch.log(_pred)
            # _loss = criterion(_pred, _labels)
            _loss = criterion(_pred, _labels)
            correct = sum(np.argmax(_pred.cpu().detach().numpy(), axis=1) == np.argmax(_label.cpu().detach().numpy(), axis=1))
            total_loss += _loss.item()
            total_counter += _pred.shape[0]
            total_correct += correct
            __pred = _pred.cpu().detach().numpy()
            __label = _label.cpu().detach().numpy()
            for idx, s_pred in enumerate(__pred):
                # print(__label[idx])
                # print(s_pred)
                # print(roc_auc_score(__label[idx], s_pred))
                Auc.append(roc_auc_score(__label[idx], s_pred))
        print("\033[0;35;40m\tEpoch {} dev average loss: {} acc {:.2%} auc {:.2}\033[0m".format(epoch, total_loss / total_counter, total_correct / total_counter, sum(Auc) / len(Auc)))
        torch.save(model.state_dict(), "large_model-{}-{:.3f}-{:.3f}-{:.2f}.pkl".format(epoch, total_loss / total_counter, total_correct / total_counter, sum(Auc) / len(Auc)))

    # # evaluate test
    # print("------------------------------Evaluate--Process--------------------------------")
    # # model.load_state_dict(torch.load('large_model....pkl'))
    # model.eval()
    # with open("prediction.txt", 'w') as f:
    #     for _idx, (_clicked_news, _candidate_news, _label) in enumerate(test_dataloader):
    #         _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.long().to(device)
    #         _pred = model(_clicked_news, _candidate_news)
    #         __pred = _pred.cpu().detach().numpy()    
    #         f.write(str(_idx + 1))
    #         f.write(" [")
    #         sortStr = [-pred for pred in __pred[0]]
    #         argStr = np.argsort(sortStr)
    #         for idx in argStr:
    #             sortStr[argStr[idx]] = 1 + idx
    #         sortStr = [str(rank) for rank in sortStr]
    #         f.write(",".join(sortStr))
    #         f.write("]")
    #         f.write("\n")
    #         if _idx % 10000 == 1:
    #             print("Line {} handled!".format(_idx))
    #     print("\033[0;35;40m\ttest prediction out!\033[0m")
    
    # evaluate devtest
    print("------------------------------Evaluate-dev-Process--------------------------------")
    # model.load_state_dict(torch.load('large_model....pkl'))
    model.eval()
    with open("dev_prediction111.txt", 'w') as f:
        for _idx, (_clicked_news, _candidate_news, _label) in enumerate(dev_test_dataloader):
            _clicked_news, _candidate_news = _clicked_news.to(device), _candidate_news.long().to(device)
            _pred = model(_clicked_news, _candidate_news)
            __pred = _pred.cpu().detach().numpy()        
            f.write(str(_idx + 1))
            f.write(" [")
            sortStr = [-pred for pred in __pred[0]]
            argStr = np.argsort(sortStr)
            for idx in argStr:
                sortStr[argStr[idx]] = 1 + idx
            sortStr = [str(rank) for rank in sortStr]
            f.write(",".join(sortStr))
            f.write("]")
            f.write("\n")
            if _idx % 10000 == 1:
                print("Line {} handled!".format(_idx))
        print("\033[0;35;40m\tdev_test prediction out!\033[0m")

    truth_file = open("dev_truth.txt", 'r')
    submission_answer_file = open("dev_prediction111.txt", 'r')
    auc, mrr, ndcg, ndcg10 = scoring(truth_file, submission_answer_file)
    print(auc)
    print(mrr)
    print(ndcg)
    print(ndcg10)

if __name__ == "__main__":
    main()
    '''advtest3 + final_data_preprocess4 + weight_decay
    Epoch 1 train average loss: 0.011395398064197588
        Epoch 1 dev average loss: 0.01173612906257016 acc 33.37% auc 0.65
    Epoch 2 train average loss: 0.0108779809379042
        Epoch 2 dev average loss: 0.0118893292221129 acc 32.76% auc 0.63
    Epoch 3 train average loss: 0.010924855955214361    
        Epoch 3 dev average loss: 0.011854545751495579 acc 33.26% auc 0.64
    Epoch 4 train average loss: 0.010906276940280407
        Epoch 4 dev average loss: 0.011883801802746218 acc 32.58% auc 0.63
    Epoch 5 train average loss: 0.01087872091134933
        Epoch 5 dev average loss: 0.011817023111795633 acc 33.70% auc 0.64
    Epoch 6 train average loss: 0.010842744859487235
        Epoch 6 dev average loss: 0.01184457434171134 acc 33.40% auc 0.64
    '''