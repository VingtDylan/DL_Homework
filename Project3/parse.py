import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=150, type=int) # 训练轮数 #3000 起飞
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=36, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=128, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=7, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--save_file', default='model/stock.pkl') # 模型保存位置
parser.add_argument('--delay', default=1, type=int) 

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device