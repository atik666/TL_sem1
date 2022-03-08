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
        transforms.Resize(256),
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

file = "ImageNet_Processed/A"

dirc = '/home/huaxia/Documents/Atik/ImageNet-ILSVRC2012/'

data_dir = dirc + file

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,
                                             shuffle=True, num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

#class_names = image_datasets['train'].classes
#img_dict = {}
#for i in range(len(class_names)):
#    img_dict[class_names[i]] = 0
#
#for i in range(len(image_datasets["train"])):
#    print(i)
#    img, label = image_datasets["train"][i]
#    img_dict[class_names[label]] += 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
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
  def __init__(self,num_classes=500):
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
      nn.Linear(in_features=4096, out_features=4096,bias=True))
    self.layer8 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Linear(in_features=4096, out_features=num_classes,bias=True))
    
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
model_ft = AlexNet()
#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Verifying CUDA
print(device)
model_ft.to(device)
print(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)

torch.save(model_ft.state_dict(), data_dir + "{}.pth".format(file[-1:]))

# new_model = TL_model('alexnet')
# new_model.load_state_dict(torch.load(dirc + "{}.pth".format(file)))

# print(new_model.eval())
