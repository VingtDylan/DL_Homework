import random

import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_data(path ="Project1/iris.data")->list:
    """加载数据集,并进行one-hot编码
    :param path: 数据集文件路径
    :return: 处理好的features和labels
    """
    headers = ['sepal_length','sepal_width','petal_length','petal_width','classes']
    iris = pd.read_csv("Project1/iris.data", names =  headers, usecols = [0, 1, 2, 3, 4])
    iris.sample(frac = 1)
    # one_hot encoder
    target_label = 'classes'
    target_class = iris[target_label]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(target_class)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    datas = iris.iloc[:,0:4].values
    labels = onehot_encoded
    return datas, labels

def train_test_validate_split(datas, labels, ratio = [0.8,0.1,0.1], random_state = 0)->list:
    """划分训练集,验证集和测试集
    :param ratio：划分比例
    :param random_state: 随机打乱数据
    :return: 划分好的训练集，验证集，测试集的featus及其标签
    """
    Datas = np.mat(datas)
    Labels = np.mat(labels)
    ratio_train = ratio[0] #训练集比例
    ratio_validate = ratio[1] #验证集比例
    ratio_test = ratio[2] #测试集比例
    assert (ratio_train + ratio_validate + ratio_test) == 1.0,'Total ratio Not equal to 1' ##检查总比例是否等于1
    cnt_train = int(len(Datas) * ratio_train)
    cnt_test = int(len(Datas) * ratio_test)
    cnt_validate = len(Datas) - cnt_train - cnt_test
    train_x = Datas[0:cnt_train]
    train_y = Labels[0:cnt_train]
    validate_x = Datas[cnt_train:cnt_train + cnt_validate]
    validate_y = Labels[cnt_train:cnt_train + cnt_validate]
    test_x = Datas[cnt_train + cnt_validate:]
    test_y = Labels[cnt_train + cnt_validate:]
    return train_x , train_y , validate_x , validate_y , test_x , test_y

def main()->None:
    # 随机种子等固定
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_path = "Project1/iris.data"
    iris_features, iris_class = load_data(file_path)
    train_x , train_y , validate_x , validate_y , test_x , test_y = train_test_validate_split(iris_features, iris_class, ratio = [0.8,0.1,0.1],random_state = 0)

    print(train_x[0])

if __name__ == '__main__':
    main()