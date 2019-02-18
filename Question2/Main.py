# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:44:36 2019

@author: Liv4dPoly
"""




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

integrated_valid_loss =[]
integrated_train_loss = []
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(in_channels = 1, out_channels = 40, kernel_size = 5, stride=1, padding = 2, dilation = 1)
        self.conv1 = nn.Conv2d(in_channels = 40, out_channels = 80, kernel_size = 5, stride=1, padding = 0, dilation = 1)
        self.conv2 = nn.Conv2d(in_channels = 80, out_channels = 50, kernel_size = 5, stride=1, padding = 0, dilation = 1)
        self.fc1 = nn.Linear(in_features = 4*4*50, out_features = 500)
        self.fc2 = nn.Linear(in_features = 500, out_features = 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    plt
def train(model, train_loader, valid_loader, optimizer, epoch, log_interval, valid_size ):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),(1-valid_size)* len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    integrated_train_loss.append(loss.item())
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data, target
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= valid_size *len(valid_loader.dataset)
    integrated_valid_loss.append(valid_loss)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        valid_loss, correct, valid_size * len(valid_loader.dataset),
        100. * correct / (valid_size * len(valid_loader.dataset))))
    
    plt.plot(range(1,1+epoch), integrated_train_loss, label="Train")
    plt.plot(range(1,1+epoch), integrated_valid_loss, label="Validation")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    batch_size = 128
    test_batch_size = 1
    epochs = 10
    lr = 0.005
    log_interval = 20
    save_model = True
    seed = 1
    
    

#    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

#    device = torch.device("cuda" if use_cuda else "cpu")

#    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
 
    from torch.utils.data.sampler import SubsetRandomSampler
    
    valid_size = 0.2
    shuffle = True
    num_train = 60000
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False,
        sampler=train_sampler)
    print(train_loader)
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False,
        sampler=valid_sampler)
    
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)


    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    
    import torchsummary
    torchsummary.summary(model, input_size=(1, 28, 28))

    for epoch in range(1, epochs + 1):
        train( model,  train_loader, valid_loader, optimizer, epoch, log_interval, valid_size)
    test( model, test_loader)

    if (save_model):
        torch.save(model,"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
