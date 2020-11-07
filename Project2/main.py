import torch

from CNN import *
from Layer import *

def main():
    # 随机种子等固定
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128 # 256 
    train_dataset = mnist.MNIST(root = './train', train = True, transform = ToTensor(), download = True)
    test_dataset = mnist.MNIST(root = './test', train = False, transform = ToTensor(), download = True)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    model = CNN().to(device)
    epoch = 10 #100
    lr = 0.002

    # optimizer = SGD(model.parameters(), lr = lr)
    optimizer = Adam(model.parameters(), lr = lr ,betas = (0.9, 0.999), eps = 1e-6)
    criterion = CrossEntropyLoss().to(device)

    model.train(device, train_loader, test_loader, epoch, optimizer, criterion)
    model.test(device, train_loader, test_loader)
    # model.save()

if __name__ == '__main__':
    main()