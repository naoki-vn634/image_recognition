import os
import torch
import argparse
import pickle

import numpy as np

from utils import *
from glob import glob

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default="/Users/matsunaganaoki/Desktop/DeepLearning/data/cifar-10-batches-py")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    data = []
    labels = []

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = Imagetransform(size,mean,std)

    dir_path = glob(os.path.join(args.input,"train/*"))
    for dir in dir_path:
        datadict = unpickle(dir)
        print(datadict[b'data'].shape)
        data.append(datadict[b"data"])
        labels.append(datadict[b"labels"])

    x_train = np.concatenate(data)
    y_train = np.concatenate(labels)
    x_test = unpickle(os.path.join(args.input,"test/test_batch"))[b"data"]
    y_test = unpickle(os.path.join(args.input,"test/test_batch"))[b"labels"]

    ##Define Dataset
    train_dataset = CIFARDataset(x_train, y_train, phase='train', transform=transform)
    val_dataset = CIFARDataset(x_test,y_test,phase='val',transform=transform)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)

    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}

    

# if __name__ == '__main__':
#     main()

import torch
from models import CustomResnet18LSTM
net = CustomResnet18LSTM()

input = torch.ones((64,10,3,224,224))
out = net(input)
