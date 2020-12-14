import random
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
from collections import Counter

from lib.utils import *
from lib.parser import args

PAD, BOS, EOS, UNK = 'PAD', 'BOS', 'EOS', 'UNK'

class DataHandler():
    def __init__(self):
        pass

    def load(self, en_path, cn_path, max_seq = 10000):
        en, cn = [], []
        s = 0
        with open(en_path, 'r', encoding='utf-8') as f:
            for line in f:
                line.strip().split('\t')
                line = word_tokenize(line.lower())
                en.append([BOS] + line + [EOS])
                s = s + 1
                if s % 100000 == 0:
                    print(s)
                if s > max_seq:
                    break
        s = 0
        with open(cn_path, 'r', encoding='utf-8') as f:
            for line in f:
                line.strip().split('\t')
                line = word_tokenize(" ".join(w for w in line))
                cn.append([BOS] + line + [EOS])
                s = s + 1
                if s % 100000 == 0:
                    print(s)
                if s > max_seq:
                    break
        for i, line in enumerate(en):
            if max(len(line), len(cn[i])) > 60:
                en.pop(i)
                cn.pop(i)
        return en, cn

    def build_dict(self, flag, kind, sentences, max_words = 1024 * 128):
        if not flag:
            print("build dict from data")
            word_count = Counter()
            for sentence in sentences:
                for s in sentence:
                    word_count[s] += 1
            ls = word_count.most_common(max_words)
            total_words = len(ls) + 2
            word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
            word_dict[PAD] = args.PAD
            word_dict[UNK] = args.UNK
            index_dict = {v: k for k, v in word_dict.items()}

            np.save(kind + '_word_dic.npy', word_dict) 
            np.save(kind + '_index_dic.npy', index_dict) 
            with open(kind + "_total_words.txt", 'w', encoding='utf-8') as f:
                f.write(str(total_words))
        else:
            print("build dict from handled file")
            word_dict = np.load(kind + '_word_dic.npy', allow_pickle = True).item()
            index_dict = np.load(kind + '_index_dic.npy', allow_pickle =True).item()
            with open(kind + "_total_words.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    line.strip().split('\t')
                    total_words = int(line)
                    # print(total_words)
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort = False):
        length = len(en)

        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
                    
        return out_en_ids, out_cn_ids

    def split_Batch(self, en, cn, batch_size, shuffle = True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]  
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
        return batches
    
    def prepare_train(self, g_max_seq1, g_max_seq2):
        self.g_max_seq1 = g_max_seq1
        self.g_max_seq2 = g_max_seq2
        self.prepare_train_data()
        self.prepare_word_dict()
        self.prepare_train_id()
        self.prepare_train_batch()

    def prepare_test(self, g_max_seq3, test_path = args.testPath):
        try:
            self.g_max_seq3 = g_max_seq3
            del self.train_en, self.valid_en
            del self.train_cn, self.valid_cn
            del self.train_en_id, self.valid_en_id
            del self.train_cn_id, self.valid_cn_id
            del self.train_data, self.valid_data
        except AttributeError:
            pass
        if not hasattr(self, 'en_word_dict'):
            self.load_word_dict()
        self.test_en, self.test_cn = self.load(test_path + "_en", test_path + "_cn", max_seq = self.g_max_seq3) 
        print("-------Test Data Loaded!------")
        self.test_en_id, self.test_cn_id = self.wordToID(self.test_en, self.test_cn, self.en_word_dict, self.cn_word_dict)
        print("----------Identify!-----------")

    def prepare_train_data(self, train_path = args.trainPath, valid_path = args.validPath):
        self.train_en, self.train_cn = self.load(train_path + "_en", train_path + "_cn", max_seq = self.g_max_seq1) 
        self.valid_en, self.valid_cn = self.load(valid_path + "_en", valid_path + "_cn", max_seq = self.g_max_seq2)
        print("-------Train Data Loaded!-----")

    def prepare_word_dict(self):
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(False, "en", self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(False, "cn", self.train_cn)
        print("------Word dicts built!-------")

    def prepare_train_id(self):
        self.train_en_id, self.train_cn_id = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.valid_en_id, self.valid_cn_id = self.wordToID(self.valid_en, self.valid_cn, self.en_word_dict, self.cn_word_dict)
        print("-----Train data identified!----")

    def prepare_train_batch(self):
        self.train_data = self.split_Batch(self.train_en_id, self.train_cn_id, args.batch_size)
        self.valid_data = self.split_Batch(self.valid_en_id, self.valid_cn_id, args.batch_size)
        print("-----Train data batched!------")

    def load_word_dict(self):
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(True, "en", None)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(True, "cn", None)
        print("------Word dicts loaded!-------")

class Batch:
    def __init__(self, src, lab = None, pad = 0):
        src = torch.from_numpy(src).to(args.device).long()
        lab = torch.from_numpy(lab).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if lab is not None:
            self.lab = lab[:, :-1]
            self.lab_y = lab[:, 1:]
            self.lab_mask = \
                self.make_std_mask(self.lab, pad)
            self.ntokens = (self.lab_y != pad).data.sum()

    @staticmethod
    def make_std_mask(lab, pad):
        lab_mask = (lab != pad).unsqueeze(-2)
        lab_mask = lab_mask & Variable(
            subsequent_mask(lab.size(-1)).type_as(lab_mask.data))
        return lab_mask