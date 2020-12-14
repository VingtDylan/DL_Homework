import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--trainPath', default = "./DataFolders/train", type = str)
parser.add_argument('--validPath', default = "./DataFolders/valid", type = str)
parser.add_argument('--testPath', default = "./DataFolders/test", type = str)
parser.add_argument('--useGPU', default = True, type = bool)
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--UNK', default = 0, type = int)
parser.add_argument('--PAD', default = 1, type = int)
parser.add_argument('--epochs', default = 1, type = int) 
parser.add_argument('--lr', default = 0.001, type = float) 
parser.add_argument('--batch_size', default = 64, type = int)
parser.add_argument('--layers', default = 2, type = int)
parser.add_argument('--h-num', default = 8, type = int) 
parser.add_argument('--batch-size', default = 64, type = int)
parser.add_argument('--d-model', default = 256, type = int) 
parser.add_argument('--d-ff', default = 1024, type = int)
parser.add_argument('--dropout', default = 0.1, type = float)
parser.add_argument('--max-length', default = 60, type = int)

args = parser.parse_args(args=['--gpu','0'])
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device