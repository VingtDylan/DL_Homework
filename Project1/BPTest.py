import random

import torch

from MyUtil import *
from DNN import *
from DNN_PYTORCH import *

def main()->None:
    # 随机种子等固定
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    file_path = "Project1/iris.data"
    iris_features, iris_class = load_data(file_path)
    train_x , train_y , validate_x , validate_y , test_x , test_y = train_test_validate_split(iris_features, iris_class, ratio = [0.8,0.1,0.1],random_state = 0)

    epochs = 1
    lr = 0.05
    # 定义layer和weight
    layer = [4,4,4,3]
    weight12 = np.random.normal(loc=0., scale=1., size=(layer[1], layer[0])) / np.sqrt(layer[0])
    weight23 = np.random.normal(loc=0., scale=1., size=(layer[2], layer[1])) 
    weight34 = np.random.normal(loc=0., scale=1., size=(layer[3], layer[2])) 
    weight = [weight12,weight23,weight34]
    
    dnn = DNN(layer = layer,weight = weight)
    dnn.train_validate(train_x, train_y, validate_x , validate_y, Epochs = epochs, batch = 1, lr = 0.05, show = False, grad_show = True)
    # dnn.test(test_x, test_y)
    
    torch.set_printoptions(precision = 5)
    net = DNN_PYTORCH(layer = layer,weight = weight)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr) 
    criterion = CEL()
    net.train_validate(train_x, train_y, validate_x , validate_y, Epochs = epochs, batch = 1, lr = 0.05, optim = optimizer, criterion = criterion, show = False, grad_show = True)
    # net.test(test_x,test_y)

if __name__ == '__main__':
    main()