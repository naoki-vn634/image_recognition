import pickle
from glob import glob
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default="/Users/matsunaganaoki/Desktop/DeepLearning/data/cifar-10-batches-py")
    args = parser.parse_args()

    data = []
    labels = []

    dir_path = glob(os.path.join(args.input,"train/*"))
    for dir in dir_path:
        datadict = unpickle(dir)
        data.append(datadict[b"data"])
        labels.append(datadict[b"labels"])

    x_train = np.concatenate(data)
    y_train = np.concatenate(labels)
    x_test = unpickle(os.path.join(args.input,"test/test_batch"))[b"data"]
    y_test = unpickle(os.path.join(args.input,"test/test_batch"))[b"labels"]
    

    #dict_keys ([b'batch_label', b'labels', b'data', b'filenames'])
    # b'batch_label : b'training batch 3 of 5'
    # b'labels' : 0-10 int
    # b'data' : ndarray (10000,3072)
    # b'filenames' :


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    main()