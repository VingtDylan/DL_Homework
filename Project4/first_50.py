import random
import os
import argparse
import math


import xlwt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import nltk
# nltk.download("all-nltk")
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter

from lib.DataLoader import *
from lib.criterion import *
from lib.loss import *
from lib.optimizer import *
from lib.parser import *
from lib.utils import *
from lib.Mytransformer import *
from evaluate import *

import copy
import warnings
warnings.filterwarnings("ignore")

def make_model(src_vocab, lab_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy
    multihead = MultiHeadedAttention(h, d_model).to(args.device)
    feedforward = FeedForward(d_model, d_ff, dropout).to(args.device)
    position = PositionalEncoding(d_model, dropout).to(args.device)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(multihead), c(feedforward), dropout).to(args.device), N).to(args.device),
        Decoder(DecoderLayer(d_model, c(multihead), c(multihead), c(feedforward), dropout).to(args.device), N).to(args.device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(args.device), c(position)),
        nn.Sequential(Embeddings(d_model, lab_vocab).to(args.device), c(position)),
        Generator(d_model, lab_vocab)).to(args.device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(args.device)

def first_evaluate(data, model):
    model.eval()
    scores1, scores2, scores3, scores4 = [], [], [], []
    t_score1, t_score2, t_score3, t_score4 = 0, 0, 0, 0
    t = len(data.test_en)
    candidates = []
    with torch.no_grad():
        for i in range(len(data.test_en)):
            reference_en = [data.en_index_dict[w] for w in data.test_en_id[i]]
            en_sent = "".join(reference_en)
            reference_cn = [data.cn_index_dict[w] for w in data.test_cn_id[i]]
            cn_sent = "".join(reference_cn)
            src = torch.from_numpy(np.array(data.test_en_id[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len = args.max_length, start_symbol = data.cn_word_dict["BOS"])
            candidate = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    candidate.append(sym)
                else:
                    break
            candidates.append("".join(candidate))
            smoothie = SmoothingFunction().method1
            score1 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1., 0., 0., 0.), smoothing_function=smoothie),4)
            score2 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./2, 1./2, 0., 0.), smoothing_function=smoothie),4)
            score3 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./3, 1./3, 1./3, 0.), smoothing_function=smoothie),4)
            score4 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./4, 1./4, 1./4, 1./4), smoothing_function=smoothie),4)
            t_score1 += score1
            t_score2 += score2
            t_score3 += score3
            t_score4 += score4
            scores1.append(score1)
            scores2.append(score2)
            scores3.append(score3)
            scores4.append(score4)
    avg_score = []
    avg_score.append(round(t_score1 / t, 4))
    avg_score.append(round(t_score2 / t, 4))
    avg_score.append(round(t_score3 / t, 4))
    avg_score.append(round(t_score4 / t, 4))
    return candidates, scores1, scores2, scores3, scores4, avg_score

def main():
    test_en_file = "./DataFolders/test_en"
    test_cn_file = "./DataFolders/test_cn"
    g_max_seq3 = 49
    en, cn = [], []
    s = 0
    with open(test_en_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            en.append(line)
            s += 1
            if s > g_max_seq3:
                break
    s = 0
    with open(test_cn_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            cn.append(line)
            s += 1
            if s > g_max_seq3:
                break

    args.layers = 2
    args.batch_size = 128
    args.d_model = 256
    args.d_ff = 1024
    args.h_num = 8
    args.dropout = 0.1
    set_seed(10)

    Data = DataHandler()
    args.src_vocab = 131074 
    args.lab_vocab = 7582 
    model = make_model(args.src_vocab, args.lab_vocab, args.layers, args.d_model, args.d_ff, args.h_num, args.dropout)
    # print(model)

    Data.prepare_test(g_max_seq3)
    model.load_state_dict(torch.load("model.pkl"))
    candidates, scores1, scores2, scores3, scores4, avg_score = first_evaluate(Data, model)
    
    workbook = xlwt.Workbook(encoding = "utf-8")
    worksheet = workbook.add_sheet("临时数据存储")
    worksheet.write(0, 0, "id")
    worksheet.write(0, 1, "label")
    worksheet.write(0, 2, "value")
    text = ["英文原文", "参考翻译", "候选翻译", "Bleu"]
    value = [en, cn, candidates]
    bleu_text = ["Bleu1 : " + str(scores1[i]) + " ; Bleu2 : " + str(scores2[i])+ " ; Bleu3 : " + str(scores3[i])+ " ; Bleu4 : " + str(scores4[i]) for i in range(g_max_seq3 + 1)]
    value.append(bleu_text)
    for idx, sample in enumerate(en):
        for i in range(4):
            if i == 0:
                worksheet.write(idx * 4 + i + 1, 0, str(idx))
            worksheet.write(idx * 4 + i + 1, 1, text[i])
            worksheet.write(idx * 4 + i + 1, 2, value[i][idx])
    workbook.save("temp.xls")
    print("平均bleu值")
    print("Bleu1 : " + str(avg_score[0]) + " ; Bleu2 : " + str(avg_score[1])+ " ; Bleu3 : " + str(avg_score[2])+ " ; Bleu4 : " + str(avg_score[3]))

if __name__ == "__main__":
    main()