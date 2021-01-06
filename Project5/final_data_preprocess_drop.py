import os
from os import path
import csv
import ast
import json
import random
import gc

from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import torch

from lib.config import nrms
from lib.utils import set_seed

def generate_stop_words(stopwords):
    stop_words = set()
    with open(stopwords, encoding = "utf-8") as f:
        for line in tqdm(f.readlines()):
            stop_words.add(line.strip('\n').strip())
    return stop_words

def news_parser(news_path, title_word_dict_path, flag = True, max_vocab_size = 50000):
    if not flag:
        print(f"Read data from handled file")
        news_id = pickle.load(open(news_parsed_path, 'rb'))
        title_word_dict = json.load(open(title_word_dict_path, 'r'))
    else:
        print(f"Parse news")
        print(f"Generate data from raw file, then save handled data to files")
        # load stopwords
        source_stopwords_path = 'stopwords.txt'
        stop_words_set = generate_stop_words(source_stopwords_path)
        news_id = {}
        title_word_dict = {nrms.pad : 0, nrms.unk : 1}
        attr_list = ['title', 'title_entities'] # ['category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities'] 
        for sample_news_path in news_path:
            # drop url
            news = pd.read_table(sample_news_path, header = None, usecols = [0, 1, 2, 3, 4, 6, 7],
                quoting = csv.QUOTE_NONE,
                names = ['id', 'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities']) 
            news.title_entities.fillna('[]', inplace = True)
            news.abstract_entities.fillna('[]', inplace = True)
            news.fillna(' ', inplace=True)
            for news_idx in tqdm(news.itertuples(), desc = "  handling with each row in news in " + sample_news_path):
                if news_idx.id not in news_id:
                    news_id[news_idx.id] = {}
                    for attr in attr_list:
                        news_id[news_idx.id][attr] = []
                    # title
                    for word in word_tokenize(news_idx.title):
                        word = word.lower()
                        if word not in stop_words_set:
                            if word not in title_word_dict:
                                title_word_dict[word] = len(title_word_dict)
                            news_id[news_idx.id]['title'].append(title_word_dict[word])
                    news_id[news_idx.id]['title'] = np.array(news_id[news_idx.id]['title'])
                    # title entity
                    title_entities_list = json.loads(news_idx.title_entities)
                    for title_entity in title_entities_list:
                        news_id[news_idx.id]['title_entities'].append(title_entity.get('WikidataId'))
                else:
                    continue
        
        with open(title_word_dict_path, 'w') as f:
            f.write(json.dumps(title_word_dict))

    print("title: {}".format(len(title_word_dict)))
    return news_id, title_word_dict

def generate_word_embedding(glove_embedding_matrix, title_word_dict = None, glove_vec_file = None, flag = False):
    if not flag:
        print(f"Read embeddings from handled file")
        used_word_embedding = pickle.load(open(glove_embedding_matrix, 'rb'))
        embedding_matrix = []
        for word in used_word_embedding:
            embedding_matrix.append(used_word_embedding[word])
    else:
        print(f"Parse glove word embedding")
        print(f"Generate word embeddings from raw file")
        word_embedding = {}
        embedding_matrix = []
        used_word_embedding = {}
        with open(glove_vec_file, encoding = 'utf-8') as f:
            for line in tqdm(f.readlines(), desc = "Handle glove pretrained file"):
                line = line.strip().split()
                val = np.array([float(item) for item in line[1:]])
                word_embedding[line[0]] = val
        for word in title_word_dict: # 8231
            if word not in word_embedding:
                init_embedding = np.random.randn(nrms.word_embedding_dim)
                scaled_init_embedding = 2 * (init_embedding - init_embedding.max()) / (init_embedding.max() - init_embedding.min()) - 1
                embedding_matrix.append(scaled_init_embedding)
                word_embedding[word] = scaled_init_embedding
                used_word_embedding[word]  = scaled_init_embedding
            else:
                embedding_matrix.append(word_embedding[word])
                used_word_embedding[word] = word_embedding[word]
        # with open(glove_embedding_matrix, encoding = 'utf-8', mode = 'w') as f:
        #     for word in word_embedding:
        #         f.write(word)
        #         f.write('\t')
        #         for val in word_embedding[word]:
        #             f.write(str(val))
        #             f.write('\t')
        #         f.write('\n')
        with open(glove_embedding_matrix, 'wb') as f:
            f.write(pickle.dumps((used_word_embedding)))
    print("used_embedding counters: {}".format(len(embedding_matrix)))
    return torch.Tensor(embedding_matrix)

def balance(clicked_news, candidate_news, label):
    positives = [i for i in range(len(label)) if label[i] == 1]
    negatives = [i for i in range(len(label)) if label[i] == 0]
    positives_len = len(positives)
    negatives_len = len(negatives)
    # print(positives_len, negatives_len)
    extra = positives_len * nrms.negative_sampling_ratio - negatives_len
    # positive samples smaller
    if extra < 0:
        negatives = negatives[:positives_len * nrms.negative_sampling_ratio]
    # negative samples smaller
    elif extra > 0:
        positives_len = negatives_len // nrms.negative_sampling_ratio
        negatives_len = positives_len * nrms.negative_sampling_ratio
        positives = positives[:positives_len]
        negatives = negatives[:positives_len * nrms.negative_sampling_ratio]
    else:
        pass
    random.shuffle(negatives)
    positives_iter, negatives_iter = iter(positives), iter(negatives)
    clicked_news_list, candidate_news_list, label_list = [], [], []
    try:
        while True:
            pos_idx = next(positives_iter)
            clicked_news_list.append(clicked_news.copy())
            candidate_news_sample_list = []
            candidate_news_sample_list.append(candidate_news[pos_idx])
            for _ in range(nrms.negative_sampling_ratio):
                neg_idx = next(negatives_iter)
                candidate_news_sample_list.append(candidate_news[neg_idx])
            candidate_news_list.append(candidate_news_sample_list)
            label_list.append([1] + [0] * nrms.negative_sampling_ratio)
            l, r = random.sample(range(0, 1 + nrms.negative_sampling_ratio), 2)
            if l == r:
                pass
            else:
                candidate_news_list[-1][l], candidate_news_list[-1][r] = candidate_news_list[-1][r], candidate_news_list[-1][l]
                label_list[-1][l], label_list[-1][r] = label_list[-1][r], label_list[-1][l]
    except StopIteration:
        pass
    return clicked_news_list, candidate_news_list, label_list

def clicked_news_structured(clicked_news):
    structured_news = []
    for news in tqdm(clicked_news, desc = "  Structure clicked news"):
        structured_title = []
        for title in news:
            if nrms.num_words_title - title.shape[0] > 0:
                paddings = np.zeros(nrms.num_words_title - title.shape[0])
                structured_title.append(np.concatenate((title, paddings)))
            else:
                structured_title.append(title[:nrms.num_words_title])
        structured_title = np.array(structured_title)
        if nrms.num_words_history - structured_title.shape[0] > 0:
            paddings = np.zeros(shape = [nrms.num_words_history - structured_title.shape[0], nrms.num_words_title])
            structured_news.append(np.concatenate((structured_title, paddings), axis=0))
        else:
            structured_news.append(structured_title[:nrms.num_words_history])
    return structured_news

def candidate_news_structured(candidate_news):
    structured_news = []
    for news in tqdm(candidate_news, desc = "  Structure candidate news"):
        structured_title = []
        for title in news:
            if nrms.num_words_title - title.shape[0] > 0:
                paddings = np.zeros(shape = [nrms.num_words_title - title.shape[0]])
                structured_title.append(np.concatenate((title, paddings), axis = 0))
            else:
                structured_title.append(title[:nrms.num_words_title])
        structured_title = np.array(structured_title)
        structured_news.append(structured_title)
    return structured_news

def behaviors_parser(news, source, target, mode):
    print(f"  Parse behaviors")
    gc.enable()
    behaviors = pd.read_table(source, sep = '\t', header = None)
    behaviors.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    behaviors.history.fillna(' ', inplace = True)
    clicked_news, candidate_news, label = [], [], []
    # train: 220w+ dev: 37w+ test: 23w+
    # TODO数据集选取
    # if mode == "train":
    #     behaviors = behaviors.iloc[:1000000] 
    # else:
    #     pass
    # behaviors = behaviors.iloc[0:1]
    # TODO balance 中以丢弃方式处理
    for row in tqdm(behaviors.itertuples(), desc="  Split user data and balance pos and neg samples"):
        t_clicked_news = []
        t_candidate_news = []
        t_label = []
        if len(row.history.split()) != 0:
            for history_news_id in row.history.split():
                if len(news[history_news_id]['title']) != 0:
                    t_clicked_news.append(news[history_news_id]['title'])
                    # print(news[history_news_id]['title'])
        else:
            #TODO 冷启动未处理
            if mode == "dev_test" or mode == "test":
                t_clicked_news.append(np.array([1]))
        if mode == "train" or mode == "dev" or mode == "dev_test":
            for impressions_news_id in row.impressions.split():
                t_candidate_news.append(news[impressions_news_id[:-2]]["title"])
                t_label.append(eval(impressions_news_id[-1]))
        elif mode == "test":
            for impressions_news_id in row.impressions.split():
                t_candidate_news.append(news[impressions_news_id]["title"])
                t_label.append(impressions_news_id)
        # 1 + K
        if mode == "train" or mode == "dev":
            if len(t_clicked_news) > 0 and len(t_candidate_news) > 0:
                t_clicked_news_list, t_candidate_news_list, t_label_list = balance(t_clicked_news, t_candidate_news, t_label)
                for sample_clicked_news in t_clicked_news_list:
                    clicked_news.append(np.array(sample_clicked_news))
                for sample_candidate_news in t_candidate_news_list:
                    candidate_news.append(np.array(sample_candidate_news))
                for sample_label in t_label_list:
                    label.append(np.array(sample_label))
        else:
            if len(t_clicked_news) > 0 and len(t_candidate_news) > 0:
                clicked_news.append(np.array(t_clicked_news))
                candidate_news.append(np.array(t_candidate_news))
                label.append(t_label)
            elif len(t_clicked_news) == 0:
                print("NotImplementedError...")
            else:
                print("Error in no candiates")
    print(f"  Structure news data and handle labels")
    # label save
    pickle.dump(label,
        open("./Processed/large_label_" + mode + "_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "wb"))
    del label
    gc.collect()

    structured_clicked_news = torch.Tensor(clicked_news_structured(clicked_news)).long()
    # structured_clicked_news save
    pickle.dump(structured_clicked_news,
        open("./Processed/large_clicked_" + mode + "_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "wb"))
    del structured_clicked_news
    gc.collect()

    if mode == "train" or mode == "dev":
        structured_candidate_news = torch.Tensor(candidate_news_structured(candidate_news)).long()
    else:
        structured_candidate_news = candidate_news_structured(candidate_news)
    # structured_candidate_news save
    pickle.dump(structured_candidate_news,
        open("./Processed/large_candidate_" + mode + "_handled_data_" + str(nrms.word_embedding_dim) + "d.txt", "wb"))
    del structured_candidate_news
    gc.collect()

def prepare_data(data_dir, news_id, mode):
    print(f'Process data for {mode}ing')
    if mode == "train":
        data_path = data_dir[0]
    elif mode == "dev" or mode == "dev_test":
        data_path = data_dir[1]
    else:
        data_path = data_dir[2]
    
    behaviors_parser(news_id, 
        path.join(data_path, 'behaviors.tsv'),
        path.join(data_path, 'behaviors_parsed.tsv'),
        mode)

def main():
    set_seed(10)
    # parse news(train and dev)
    data_dir = ['./data/train', './data/dev', './data/test']
    title_word_dict_path = path.join('./Processed', 'large_title_word_dict.json')

    news_id, title_word_dict= \
        news_parser([path.join(data_dir[0], 'news.tsv'), path.join(data_dir[1], 'news.tsv'), path.join(data_dir[2], 'news.tsv')], 
        title_word_dict_path,
        flag = True) # True for regenerate

    # embedding_matrix
    word_vec_file = path.join('glove.6B', 'glove.6B.' + str(nrms.word_embedding_dim) + 'd.txt')
    glove_embedding_matrix_path = './Processed/large_glove_embedding_matrix_' + str(nrms.word_embedding_dim) + 'd.txt'
    embedding_matrix = generate_word_embedding(glove_embedding_matrix_path, title_word_dict, word_vec_file, flag = True)

    prepare_data(data_dir, news_id, "train") 

    prepare_data(data_dir, news_id, "dev")

    prepare_data(data_dir, news_id, "test")

    prepare_data(data_dir, news_id, "dev_test")

if __name__ == "__main__":
    main()