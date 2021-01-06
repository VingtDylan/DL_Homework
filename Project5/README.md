固定的学习率:

* AUC 0.6689
* MRR 0.3206
* NDCG5 0.3536
* NDCG10 0.4175

动态的学习率:

* AUC 0.6782
* MRR 0.3278
* NDCG5 0.3616
* NDCG10 0.4254

```
News Recommendation
├─ data(数据集)
│  ├─ dev(验证集)
│  │  ├─ behaviors.tsv
│  │  ├─ entity_embedding.vec
│  │  ├─ news.tsv
│  │  └─ relation_embedding.vec
│  ├─ test(测试集)
│  │  ├─ behaviors.tsv
│  │  ├─ entity_embedding.vec
│  │  ├─ news.tsv
│  │  └─ relation_embedding.vec
│  └─ train(训练集)
│     ├─ behaviors.tsv
│     ├─ entity_embedding.vec
│     ├─ news.tsv
│     └─ relation_embedding.vec
├─ dev_evaluate.py(验证集评估函数文件)
├─ dev_prediction.txt(验证集排序结果)
├─ dev_scores.txt(验证集评估结果)
├─ dev_truth.txt(验证集真值标签)
├─ evaluate.py(评估函数文件)
├─ final_data_preprocess_drop.py(数据集处理文件)
├─ glove.6B
│  └─ glove.6B.100d.txt(glove预训练词向量)
├─ large_model-9-0.011-0.397-0.69.pkl(示例模型文件)
├─ lib(自定义的一些功能文件)
│  ├─ config.py
│  ├─ dataset.py
│  └─ utils.py
├─ main.py(训练主文件)
├─ model(自定义模型文件)
│  ├─ AttentionNetWork.py
│  ├─ MultiHeadSelfAttention.py
│  └─ NRMS.py
├─ prediction.txt(测试集预测结果)
├─ prediction_dev.py(生成测试集集排序结果的函数文件)
├─ prediction_test.py(生成验证集排序结果的函数文件)
├─ Processed(用于存储生成的文件)
│  └─ 后续生成的文件
├─ statistics.py (数据分析文件)
└─ stopwords.txt (停止词文件)
```