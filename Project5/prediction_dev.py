
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
from lib.dataset import MyDataSet
from lib.utils import set_seed
from lib.config import nrms
from final_data_preprocess_drop import generate_word_embedding

def main():
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

    print(f"Load dev_test data")
    dev_test_clicked_news = \
        pickle.load(open("./Processed/large_clicked_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_candidate_news = \
        pickle.load(open("./Processed/large_candidate_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
    dev_test_label = \
        pickle.load(open("./Processed/large_label_dev_test_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "rb"))
        
    print("dev_test data counters: {}".format(len(dev_test_label)))

    dev_test_dataset = MyDataSet(dev_test_clicked_news, dev_test_candidate_news, dev_test_label)
    dev_test_dataloader = DataLoader(dev_test_dataset, batch_size = 1, shuffle = False)

    # evaluate devtest 
    print("------------------------------Evaluate-dev-Process--------------------------------")
    model.load_state_dict(torch.load('large_model-0.668.pkl'))
    model.eval()
    with open("dev_prediction.txt", 'w') as f:
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

if __name__ == "__main__":
    main()