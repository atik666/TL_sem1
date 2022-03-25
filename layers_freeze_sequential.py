from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import copy

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
<<<<<<< HEAD
        #transforms.Resize(256),
=======
        transforms.Resize(256),
>>>>>>> 60267ff (linux)
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

<<<<<<< HEAD
weight_file = "B"
num_layer = 4
data_file = "B"

dirc = '/home/huaxia/Documents/Atik/'
=======
weight_file = "BB100"
num_layer = 1
data_file = "B"

print("Model: %s%d%s" %(weight_file, num_layer, data_file))

dirc = '/home/admin1/Documents/Atik/Imagenet/partitioned/'
>>>>>>> 60267ff (linux)

data_dir = dirc + data_file

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

<<<<<<< HEAD
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
=======
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
>>>>>>> 60267ff (linux)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class AlexNet(nn.Module):
<<<<<<< HEAD
  def __init__(self,num_classes=50):
=======
  def __init__(self,num_classes=500):
>>>>>>> 60267ff (linux)
    super(AlexNet,self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2,padding=0))
    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=64,out_channels=192,kernel_size=5,stride=1,padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3,stride=2,padding=0))
    self.layer3 = nn.Sequential(
      nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1),
      nn.ReLU(inplace=True))
    self.layer4 = nn.Sequential(
      nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
      nn.ReLU(inplace=True))
    self.layer5 = nn.Sequential(
      nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
    self.layer6 = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=256*6*6,out_features=4096,bias=True),
      nn.ReLU(inplace=True))
    self.layer7 = nn.Sequential(
      nn.Dropout(p=0.5),
<<<<<<< HEAD
      nn.Linear(in_features=4096, out_features=2048,bias=True))
    self.layer8 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Linear(in_features=2048, out_features=num_classes,bias=True))
=======
      nn.Linear(in_features=4096, out_features=4096,bias=True))
    self.layer8 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=num_classes,bias=True))
>>>>>>> 60267ff (linux)
    
  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = x.view(x.size(0),256*6*6)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    return x

#Move the input and AlexNet_model to GPU for speed if available
new_model = AlexNet()
#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Verifying CUDA
print(device)
new_model.to(device)
print(new_model)

<<<<<<< HEAD
new_model.load_state_dict(torch.load(dirc + "layers_{}.pth".format(weight_file)))
=======
new_model.load_state_dict(torch.load(dirc + "{}.pth".format(weight_file)))
>>>>>>> 60267ff (linux)

print(new_model.eval())

def freeze_layers(new_model, layer):
    count = 0
    for param in new_model.parameters():
        count +=1
        if count < 2 * layer + 1: #freezing first n layers
            param.requires_grad = False
            
    for name, param in new_model.named_parameters():
        print(name, ':', param.requires_grad)
        
    return new_model

new_model = freeze_layers(new_model, layer = num_layer)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)          
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

new_model.apply(weight_init)    

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
<<<<<<< HEAD
optimizer_ft = optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)
=======
optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)
>>>>>>> 60267ff (linux)

# Decay LR by a factor
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)
            
new_model = train_model(new_model, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs=100)
        
<<<<<<< HEAD

=======
print("Model: %s%d%s" %(weight_file, num_layer, data_file))
>>>>>>> 60267ff (linux)
