from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../dataset/birds/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_fn_kd(outputs, labels, teacher_outputs, params, inputs, epoch):
    alpha = params['alpha']
    T = params['temperature']
    if epoch < 35:
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels) * (1. - alpha)
    else:
        KD_loss = F.cross_entropy(outputs, labels)
    return KD_loss

def plot_losses_accuracies(losses, accuracies):
    plt.plot(range(num_epochs), losses['train'])
    plt.plot(range(num_epochs), losses['val'])
    plt.title('Loss statistics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Validation loss'])
    plt.savefig('losses.png')
    plt.close()

    plt.plot(range(num_epochs), accuracies['train'])
    plt.plot(range(num_epochs), accuracies['val'])
    plt.title('Accuracy statistics')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train acc', 'Validation acc'])
    plt.savefig('accuracies.png')
    plt.close()

def plot_learning_rate(learning_rate):
    plt.plot(range(num_epochs), learning_rate)
    plt.title('Learning rate over time')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.savefig('lr.png')
    plt.close()

def train_model(model, model_teacher, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    learning_rate = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    params = {'alpha': 0.1, 'temperature': 6}
    model_teacher.eval()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    outputs_teacher = model_teacher(inputs)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = loss_fn_kd(outputs, labels, outputs_teacher, params, inputs, epoch)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                
                running_corrects += torch.sum(preds == labels.data)
            

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
                learning_rate.append(scheduler.get_last_lr())
                losses['train'].append(epoch_loss)
                accuracies['train'].append(epoch_acc)

            if phase == 'val':
                losses['val'].append(epoch_loss)
                accuracies['val'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

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
    return model, losses, accuracies, learning_rate

model_ft = models.resnext101_32x8d(pretrained=True)
model_teacher = models.resnext101_32x8d()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)
model_teacher.fc = nn.Linear(num_ftrs, 200)

model_ft = model_ft.to(device)
model_teacher = model_teacher.to(device)
model_teacher.load_state_dict(torch.load('weights/best_weight_decay_4e-5.pt'))
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-4)
num_epochs = 60

exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10, eta_min=0)
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)
model_ft, losses, accuracies, learning_rate = train_model(model_ft, model_teacher, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)

plot_losses_accuracies(losses, accuracies)
plot_learning_rate(learning_rate)
torch.save(model_ft.state_dict(), 'weights/best.pt')



