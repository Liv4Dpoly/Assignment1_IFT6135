import torchvision
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import csv
import os
import os.path
from PIL import Image

# Training on GPU
# ----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def load_train_images(root="./ift6135h19/trainset/trainset/", batch_size=32, valid_size = 0.2, shuffle = True, num_train = 19998):
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))   
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)    
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # Data augmentation
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomRotation(40),
            transforms.ToTensor()])
            
      
    train_set = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               sampler=train_sampler,
                                               num_workers=0)
    
    transform = transforms.Compose([transforms.ToTensor()])
    valid_set = torchvision.datasets.ImageFolder(root=root, transform=transform)
    valid_loader = torch.utils.data.DataLoader(valid_set, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               sampler=valid_sampler,
                                               num_workers=0)

    return train_loader, valid_loader


#Return dataset with file paths 
class ImageFolderGetPath(datasets.ImageFolder):
    # override the __getitem__ method. 
    def __getitem__(self, index):
        original_tuple = super(ImageFolderGetPath, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_w_path = (original_tuple + (path,))
        return tuple_w_path
# Load test images with their full filename  
def load_test_images(root="./ift6135h19/testset"):
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = ImageFolderGetPath(root=root, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0)
    return test_loader

integrated_valid_loss =[]
integrated_train_loss = []
integrated_valid_error =[]
integrated_train_error = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 50, 5, 1)
        self.fc1 = nn.Linear(13*13*50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 13*13*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(model, train_loader, valid_loader, optimizer, epoch, log_interval, valid_size ):
    model = model.to(device)
    #Training
    model.train()
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        data,target = data.to(device),target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target).to(device)
        training_pred = output.argmax(dim=1, keepdim=True).to(device) # get the index of the max log-probability
        train_correct += training_pred.eq(target.view_as(training_pred)).sum().item()
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
    train_correct = train_correct / ((1-valid_size) * len(train_loader.dataset))
    train_error = 1 - train_correct
    integrated_train_loss.append(loss.item())
    integrated_train_error.append(train_error)

    #Validation
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True).to(device) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= valid_size *len(valid_loader.dataset)
    integrated_valid_loss.append(valid_loss)
    valid_correct = correct / (valid_size * len(valid_loader.dataset))
    valid_error = 1 - valid_correct
    integrated_valid_error.append(valid_error)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        valid_loss, correct, int(np.floor((valid_size * len(valid_loader.dataset)))),
        valid_correct))
    
    # plot training process 
    if epoch % 60 == 0:
        plt.plot(range(1,1+epoch), integrated_train_loss, label="Train Loss")
        plt.plot(range(1,1+epoch), integrated_valid_loss, label="Validation Loss")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

        plt.plot(range(1,1+epoch), integrated_train_error, label="Train Error")
        plt.plot(range(1,1+epoch), integrated_valid_error, label="Validation Error")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
    
    
    
def test(model, test_loader):
    model = model.to(device)
    labels_ = ( 'Cat', 'Dog')
    csv_map = {}

    with torch.no_grad():
        for  images,_, filepath in test_loader:

            filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
            filepath = int(filepath)
            images = images.to(device)
            outputs = model(images)
            predicted = torch.max(outputs.data, 1)[1].to(device).data
            predicted = predicted.item()
            csv_map[filepath] = labels_[predicted]

        with open('./prediction.csv', 'w') as csvfile:
            fieldnames = ['id', 'label']
            csv_w = csv.writer(csvfile)
            csv_w.writerow(('id', 'label'))
        
            for row in sorted(csv_map.items()):
                csv_w.writerow(row)
        
#Hyper-parameters 
batch_size = 32            
epochs = 60
lr = 0.03
log_interval = 20
save_model = False
seed = 1
valid_size = 0.2
    
#Train and validate the model
(train_loader, valid_loader) = load_train_images(batch_size = batch_size)
model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    train( model,  train_loader, valid_loader, optimizer, epoch, log_interval, valid_size)
torch.save(model,"Cat-Dog_cnn.pt")
#Test the trained model
test_loader = load_test_images()
test( model, test_loader)

#Display some (batch_size) validation images and check the output probability from the model

dataiter = iter(valid_loader)
sample_images, sample_labels = dataiter.next()
img = torchvision.utils.make_grid(sample_images[0:batch_size])
imgnp = img.numpy()
plt.imshow(np.transpose(imgnp, (1, 2, 0)))
plt.show()   

sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
classes = ( 'Cat', 'Dog')
outputs = model(sample_images)
_, predicted = torch.max(outputs, 1)
output_probability = torch.exp(outputs)
output_probability = torch.round(output_probability* 100) 
for j in range(0,batch_size):
    if (sample_labels[j] != predicted [j]):
        print('Img idx: {}  GroundTruth: {}       Predicted: {}        Probability[Cat Dog]: {} '.format(j, classes[sample_labels[j]], classes[predicted[j]],  output_probability[j].tolist()))
        

