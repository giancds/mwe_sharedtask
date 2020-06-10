import os
import pickle

import numpy as np
import torch
from torchtext.data import Dataset, Field, Example
from torchtext.data.iterator import BucketIterator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



def load_tokenized_data(datafile, language_codes, val_size=0.15, seed=42):

    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = [], []
    x_dev, y_dev = {}, {}
    for code in language_codes:

        true_x, true_y = [], []
        false_x, false_y = [], []
        for xsample, ysample in zip(data[code]['x_train'], data[code]['y_train']):
            if 1 in ysample:
                true_x.append(xsample)
                true_y.append(ysample)


        max_len = max([len(y) for y in true_y])
        print(max_len)

        for xsample, ysample in zip(data[code]['x_train'], data[code]['y_train']):
            if 1 not in ysample and len(ysample) < max_len:
                false_x.append(xsample)
                false_y.append(ysample)
        false_x = np.array(false_x)
        false_y = np.array(false_y)

        np.random.seed(seed)
        idx = np.random.randint(len(false_y), size=len(true_y))
        false_x = false_x[idx].tolist()
        false_y = false_y[idx].tolist()

        x_train += true_x + false_x
        y_train += true_y + false_y

        x_dev[code] = data[code]["x_dev"]
        y_dev[code] = data[code]["y_dev"]

    del data

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=seed)

    return (x_train, y_train),( x_val, y_val), (x_dev, y_dev)

def load_and_tokenize_dataset(train_files, tokenizer, train=True):

    if len(train_files) == 0:
        files = []
        for root, _, files in os.walk('data/'):
            for _file in files:
                if train:
                    if _file == 'train.cupt':
                        files.append(os.path.join(root, _file))
                else:
                    if _file == 'dev.cupt':
                        files.append(os.path.join(root, _file))

    else:
        files = train_files

    for _file in files:
        print(_file)
    cls = tokenizer.encode('[CLS]')[1]
    sep = tokenizer.encode('[SEP]')[1]

    sentences, labels = [], []
    for _file in files:
        with open(_file) as text:
            tmp_line = []
            tmp_label = []
            for line in text:
                if line == '\n':
                    sentences.append([cls] + tmp_line + [sep])
                    labels.append([0] + tmp_label + [0])
                    tmp_line = []
                    tmp_label = []
                elif not line.startswith('#'):
                    feats = line.split()
                    _label = 0 if feats[10] == '*' else 1
                    tokens = tokenizer.encode(feats[1])
                    tokens = tokens[1:-1]
                    _label = [_label] * len(tokens)
                    tmp_line += tokens
                    tmp_label += _label
    return sentences, labels

