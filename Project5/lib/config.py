import os

class BaseConfig():
    device = 0
    num_epochs = 10
    batch_size = 128
    num_workers = 4
    learning_rate = 0.001 # 0.0002
    dropout = 0.2

    pad = '<pad>'
    unk = '<unk>'
    pad_idx = 0
    unk_idx = 1
    negative_sampling_ratio = 4 # K = 4
    num_words = 57478 # 41342 small 57478 large
    num_words_title = 15 # 48
    num_words_history = 20 # 50

    entity_embedding_dim = 100 # 200 
    word_embedding_dim =  100 # 300 Glove embedding
    category_vec_dim = 100
    query_vector_dim = 200

class NRMSConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    num_attention_heads = 20

nrms = NRMSConfig()