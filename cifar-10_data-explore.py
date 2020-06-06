# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:56:25 2020

@author: Michael Chen
"""


import torch, sys, os
import torch.nn as nn
import torchvision 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd

train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = torchvision.transforms.ToTensor()


BATCH_SIZE = 1
TRAIN_COUNT = 40_000
VAL_COUNT = 10_000
TEST_COUNT = 10_000

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
class_label = {0:'plane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
ROOT = r'C:\Users\user\Documents\proposal'
LIST_csv_name = ['cifar-10-train-.csv','cifar-10-val-.csv','cifar-10-test-.csv']


def Dataset2csv():
    BATCH_SIZE = 1
    TRAIN_COUNT = 40_000
    VAL_COUNT = 10_000
    TEST_COUNT = 10_000
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_set = torch.utils.data.dataset.Subset(train_set, range(0,TRAIN_COUNT))
    trainLoader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    val_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    val_set = torch.utils.data.dataset.Subset(val_set, range(TRAIN_COUNT,TRAIN_COUNT+VAL_COUNT))
    valLoader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testLoader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    with open(os.path.join(ROOT,LIST_csv_name[0]),'w') as f:
        f.write('idx,')
        f.write('category\n')
        for idx, (img, category) in enumerate(trainLoader):
            category =  str(category).split('tensor')[1].split('[')[1].split(']')[0]
            f.write(str(idx)+','+str(category)+'\n')
            
    
    
    with open(os.path.join(ROOT,LIST_csv_name[1]),'w') as f:
        f.write('idx,')
        f.write('category\n')
        for idx, (img, category) in enumerate(valLoader):
            category =  str(category).split('tensor')[1].split('[')[1].split(']')[0]
            f.write(str(idx)+','+str(category)+'\n')
            
            
            
    with open(os.path.join(ROOT,LIST_csv_name[2]),'w') as f:
        f.write('idx,')
        f.write('category\n')
        for idx, (img, category) in enumerate(testLoader):
            category =  str(category).split('tensor')[1].split('[')[1].split(']')[0]
            f.write(str(idx)+','+str(category)+'\n')        


#Dataset2csv()



for item in LIST_csv_name:    
    csv_df = None
    csv_df = pd.read_csv(os.path.join(ROOT,item))
    category_conut = len(csv_df.category.unique())
    item_count = csv_df.shape[0]
    plt.rcParams.update({'font.size': 22})    
    temp = csv_df.groupby('category').count()
    count = temp.iloc[:,0].tolist()
    plt.figure(figsize=(16,9))
    plt.title("Cifar-10 Category Histogram "+item.split('-')[2])
    plt.title("Cifar-10 Category Histogram "+item.split('-')[2])
    plt.bar(class_label.values(), count, alpha=0.9, color=color_cycle)
    for x, y in zip(class_label.values(), count):
        plt.text(x, y*3/4, str(y), fontsize=25, color='white', ha='center')
    plt.show()
