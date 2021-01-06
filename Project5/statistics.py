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
from matplotlib import pyplot as plt

from lib.config import nrms
from lib.utils import set_seed

def generate_stop_words(stopwords):
    stop_words = set()
    with open(stopwords, encoding = "utf-8") as f:
        for line in tqdm(f.readlines()):
            stop_words.add(line.strip('\n').strip())
    return stop_words

def news_parser(news_path, max_vocab_size = 50000):
    print(f"Parse news")
    print(f"Generate data from raw file, then save handled data to files")
    # load stopwords
    source_stopwords_path = 'stopwords.txt'
    stop_words_set = generate_stop_words(source_stopwords_path)
    news_id = {}
    attr_list = ['title'] 
    title_word_dict = {nrms.pad : 0, nrms.unk : 1}
    title_length = {}
    title_total_length = 0
    max_title_length = 0
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
                length = len(news_id[news_idx.id]['title'])
                if length <= 0:
                    pass
                elif length not in title_length:
                    title_length[length] = 1
                else:
                    title_length[length] += 1
                title_total_length += length
                if length > max_title_length:
                    max_title_length = length
                news_id[news_idx.id]['title'] = np.array(news_id[news_idx.id]['title'])
            else:
                continue
    print("news: {}, avg_title: {:.3}, max_title_length: {}".format(len(news_id), title_total_length / (len(news_id)), max_title_length))
    
    X = range(1, 1 + max_title_length)
    Y = []
    plt.figure()
    for _length in X:
        Y.append(title_length.get(_length, 0))
    plt.bar(X, Y, width = 0.5)
    plt.show()

def behaviors_parser(source):
    print(f"  Parse behaviors")
    behaviors = pd.read_table(source, sep = '\t', header = None)
    behaviors.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    behaviors.history.fillna(' ', inplace = True)
    user = {}
    positives_samples = 0
    negatives_samples = 0
    for row in tqdm(behaviors.itertuples()):
        if row.user_id not in user:
            user[row.user_id] = len(user) + 1
        for impressions_news_id in row.impressions.split():
            value = impressions_news_id[-1]
            if value == "0":
                negatives_samples += 1
            else:
                positives_samples += 1
    print("user: {}, positives: {}, negatives: {}".format(len(user), positives_samples, negatives_samples))

def main():
    set_seed(10)

    news_dir = ['./data/train/news.tsv']
    title_word_dict_path = 'large_title_word_dict.json'

    news_parser(news_dir, title_word_dict_path)

    behaviors_dir = ['./data/train/behaviors.tsv']
    behaviors_parser(behaviors_dir[0])

if __name__ == "__main__":
    main()