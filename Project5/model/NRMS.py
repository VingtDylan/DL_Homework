import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, softmax, log_softmax

from model.MultiHeadSelfAttention import MultiHeadSelfAttention
from model.AttentionNetWork import AttentionNetWork
from lib.config import nrms

device = torch.device(f"cuda:{nrms.device}" if torch.cuda.is_available() else "cpu")

class NRMS(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding = None):
        super(NRMS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = ClickPredictor()

    def forward(self, clicked_news, candidate_news):  
        # print(clicked_news.shape)
        # print(candidate_news.shape)
        candidate_news_vector = torch.stack([self.news_encoder(x) for x in candidate_news], dim = 0)
        # print(candidate_news_vector.shape)
        clicked_news_vector = torch.stack([self.news_encoder(x) for x in clicked_news], dim = 0)
        # print(clicked_news_vector.shape)
        user_vector = self.user_encoder(clicked_news_vector)
        # print(user_vector.shape)
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        # print(click_probability.shape)
        return click_probability
        # return click_probability + torch.Tensor([1e-8]).to(device)
        # return sigmoid(click_probability)
        # return  softmax(click_probability, dim = 1)
        
class NewsEncoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding = None):
        '''
        layer 1： word embedding layer
        layer 2:  word_level multi-head self-attention network
        layer 3:  word attention network 
        '''
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words, config.word_embedding_dim, padding_idx = 0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze = False, padding_idx = 0)
        self.multihead_self_attention = MultiHeadSelfAttention(config.word_embedding_dim, config.num_attention_heads)
        self.attention_network = AttentionNetWork(config.query_vector_dim, config.word_embedding_dim)
    
    def forward(self, news):
        '''
        input news
        '''
        # print(news.shape)
        news_vector = F.dropout(self.word_embedding(news), p = self.config.dropout)
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector, p = self.config.dropout)
        f_news_vector = self.attention_network(multihead_news_vector)
        # print(f_news_vector.shape)
        return f_news_vector

class UserEncoder(nn.Module):
    def __init__(self, config):
        '''
        layer 1: new-level multi-head self-attention network
        layer 2: news attention network
        '''
        super(UserEncoder, self).__init__()
        self.config = config
        self.multihead_self_attention = MultiHeadSelfAttention(config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AttentionNetWork(config.query_vector_dim, config.word_embedding_dim)

    def forward(self, user_vector):
        multihead_user_vector = self.multihead_self_attention(user_vector)
        f_user_vector = self.additive_attention(multihead_user_vector)
        return f_user_vector

class ClickPredictor(nn.Module):
    def __init__(self):
        '''
        inner product
        '''
        super(ClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        # print("------------------------------111111111111")
        # print(candidate_news_vector.shape)
        # print(user_vector.shape)
        probability = torch.bmm(candidate_news_vector,user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        # print(probability.shape)
        return probability