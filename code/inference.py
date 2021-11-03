from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import pathlib
import os
import copy
from PIL import Image


def get_class_from_label(class_names, label):
    return class_names[list(range(0, 200)).index(label)]

def read_class_names(path):
    class_names = []
    with open(path, 'r') as class_names_file:
        for line in class_names_file:
            class_names.append(line)
    return class_names

data_transforms = {
    'train': transforms.Compose([
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


class_names = read_class_names('./class_names.txt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnext101_32x8d(pretrained=False)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 200)

model_ft.load_state_dict(torch.load('weights/best_kd_weight_decay_4e-5.pt'))

model_ft = model_ft.to(device)
model_ft.eval()
predictions = []
transform = data_transforms['val']
names = []
answer_order = []
final_predictions = []

with open('answer-example.txt', 'r') as answer_example:
    for line in answer_example:
        answer_order.append(line.split(' ')[0])

for image_path in pathlib.Path('../dataset/birds/data/inference').iterdir():
    image = Image.open(image_path)
    print(image_path.name)
    input = transform(image)
    input = input.unsqueeze(0).to(device)
    outputs = model_ft(input)
    _, preds = torch.max(outputs.data, 1)
    class_name = get_class_from_label(class_names, preds.item())
    names.append(image_path.name)
    predictions.append(f'{image_path.name} {class_name}')

for ans in answer_order:
    idx = names.index(ans)
    final_predictions.append(predictions[idx])

with open('answer.txt', 'w') as submission:
    submission.writelines(final_predictions)



