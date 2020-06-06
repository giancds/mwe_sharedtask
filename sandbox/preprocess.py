
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from enum import Enum
from sklearn.model_selection import train_test_split


class Features(Enum):
    word = 1
    upos = 3
    xpos = 4
    deprel = 7


def process_cup(text):

    id_sent = None
    features = []

    for line in text:

        if line is '\n':
            id_sent = None
        elif line.startswith('# source_sent_id'):
            tokens = line.split()
            id_sent = tokens[-1]
        elif not line.startswith('#'):
            feats = line.split()

            feats_dict = {
                'id_sent': id_sent,
                'id': feats[0],
                'form': feats[1],
                'lemma': feats[2],
                'upos': feats[3],
                'xpos': feats[4],
                'feats': feats[5],
                'head': feats[6],
                'deprel': feats[7],
                'deps': feats[8],
                'misc': feats[9],
                'mwe': feats[10]
            }
            features.append(feats_dict)

    return pd.DataFrame(features)

def load_tokenized_data(datafile, language_codes):

    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = [], []
    x_dev, y_dev = [], []
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

        np.random.seed(SEED)
        idx = np.random.randint(len(false_y), size=len(true_y))
        false_x = false_x[idx].tolist()
        false_y = false_y[idx].tolist()

        x_train += true_x + false_x
        y_train += true_y + false_y

        x_dev += data[code]["x_dev"]
        y_dev += data[code]["y_dev"]



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
                    _label = 0 if feats[10] is '*' else 1
                    tokens = tokenizer.encode(feats[1])
                    tokens = tokens[1:-1]
                    _label = [_label] * len(tokens)
                    tmp_line += tokens
                    tmp_label += _label
    return sentences, labels


def load_word_dataset(train_files, train=True):

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

    sentences, labels = [], []
    for _file in files:
        with open(_file) as text:
            tmp_line = None
            flag = False
            for line in text:
                if line == '\n':
                    sentences.append(tmp_line)
                    labels.append(1 if flag else 0)
                    tmp_line = None
                    flag = False
                elif line.startswith('# text = '):
                    tmp_line = line.replace('# text = ', '').replace('\n', '')
                elif not line.startswith('#'):
                    feats = line.split()
                    if feats[10] is not '*' and not flag:
                        flag = True
    return sentences, labels

def extract_dataset(files, per_word=False, feature=Features.upos):
    data = []
    processing_func = _build_per_word_dataset if per_word else _build_dataset
    for file in files:
        with open(file) as f:
            data += processing_func(f, feature=feature)
    return data


def _build_dataset(text, feature=Features.upos):
    examples = []
    flag = False
    example = ''
    for line in text:
        if line is '\n':     # if it is an empty line, we reset everything
            label = 1 if flag else 0
            examples.append((example.strip(), label))
            example = ''
            flag = False

        elif not line.startswith('#'):     # if it is not a line of metadata
            feats = line.split()
            example += ' ' + feats[feature.value]
            if feats[10] is not '*' and flag == False:
                flag = True

    return examples


def _build_per_word_dataset(text, feature=Features.upos):
    examples = []
    example = ''
    labels = []
    for line in text:
        if line is '\n':     # if it is an empty line, we reset everything
            examples.append((example.strip(), labels))
            example = ''
            labels = []

        elif not line.startswith('#'):     # if it is not a line of metadata
            feats = line.split()
            example += ' ' + feats[feature.value]
            label = 0 if feats[10] is '*' else 1
            labels.append(label)

    return examples


def load_dataset(train_files, feature, per_word=False, cnn=False, train=True):

    if cnn:
        return _load_dataset_for_cnn(train_files, feature, per_word, train)

    else:
        return _load_dataset_for_rnn(train_files, feature, per_word, train)


def _load_dataset_for_cnn(train_files, features, per_word=False, train=True):

    if len(train_files) == 0:
        _train_files = []
        for root, _, files in os.walk('data/'):
            for _file in files:
                if train:
                    if _file == 'train.cupt':
                        _train_files.append(os.path.join(root, _file))
                else:
                    if _file == 'dev.cupt':
                        _train_files.append(os.path.join(root, _file))

    else:
        _train_files = train_files

    train_sents, train_labels = [], []
    for i, feature in enumerate(features):
        tmp = extract_dataset(_train_files, per_word=per_word, feature=feature)
        if i == 0:
            train_labels = [d[1] for d in tmp]
        _train_sents = [d[0] for d in tmp]
        train_sents.append(_train_sents)

    return train_sents, train_labels


def _load_dataset_for_rnn(train_files, feature, per_word=False, train=True):

    if len(train_files) == 0:
        _train_files = []
        for root, _, files in os.walk('data/'):
            for file in files:
                if train:
                    if file == 'train.cupt':
                        _train_files.append(os.path.join(root, file))
                else:
                    if file == 'dev.cupt':
                        _train_files.append(os.path.join(root, file))
    else:
        _train_files = train_files

    train_dataset = extract_dataset(_train_files,
                                    per_word=per_word,
                                    feature=feature)

    sents = [d[0] for d in train_dataset]
    labels = [d[1] for d in train_dataset]

    return sents, labels


def pre_process_data(train_data,
                     dev_data,
                     test_size=0.15,
                     seed=None,
                     pad_targets=False,
                     cnn=False):
    if cnn:
        return _pre_process_for_cnn(train_data, dev_data, test_size, seed,
                                    pad_targets)
    else:
        return _pre_process_for_rnn(train_data, dev_data, test_size, seed,
                                    pad_targets)


def _pre_process_for_cnn(train_data,
                         dev_data,
                         test_size=0.15,
                         seed=None,
                         pad_targets=False):
    train_sents, train_labels = train_data
    dev_sents, dev_labels = dev_data

    tokenizers = []
    x_train = []
    for dataset in train_sents:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ', filters='')
        tokenizer.fit_on_texts(dataset)
        _x_train = tokenizer.texts_to_sequences(dataset)
        _x_train = tf.keras.preprocessing.sequence.pad_sequences(
            _x_train)     # pad to the longest sequence length
        tokenizers.append(tokenizer)
        x_train.append(np.expand_dims(np.array(_x_train), axis=2))

    x_train = np.concatenate(x_train, axis=2)

    max_len = x_train.shape[1]

    if pad_targets:
        y_train = tf.keras.preprocessing.sequence.pad_sequences(train_labels,
                                                                maxlen=max_len,
                                                                value=-1.0,
                                                                padding='post')
        y_dev = tf.keras.preprocessing.sequence.pad_sequences(dev_labels,
                                                              maxlen=max_len,
                                                              value=-1.0,
                                                              padding='post')

    else:
        y_train = np.array(train_labels).reshape(-1, 1)
        y_dev = np.array(dev_labels).reshape(-1, 1)

    y_train = np.array(train_labels).reshape(-1, 1)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.15,
                                                      random_state=seed)

    x_dev = []
    for i, dataset in enumerate(dev_sents):
        _x_dev = tokenizer.texts_to_sequences(dataset)
        _x_dev = tf.keras.preprocessing.sequence.pad_sequences(_x_dev,
                                                               maxlen=max_len)
        x_dev.append(np.expand_dims(np.array(_x_dev), axis=2))

    x_dev = np.concatenate(x_dev, axis=2)

    return (x_train, x_val, y_train,
            y_val), (x_dev,
                     y_dev), (max_len,
                              max([len(t.word_index) for t in tokenizers]))


def _pre_process_for_rnn(train_data,
                         dev_data,
                         test_size=0.15,
                         seed=None,
                         pad_targets=False):
    train_sents, train_labels = train_data
    dev_sents, dev_labels = dev_data

    tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ', filters='')
    tokenizer.fit_on_texts(train_sents)

    x_train = tokenizer.texts_to_sequences(train_sents)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train)     # pad to the longest sequence length

    max_len = x_train.shape[1]
    if pad_targets:
        y_train = tf.keras.preprocessing.sequence.pad_sequences(train_labels,
                                                                maxlen=max_len,
                                                                value=-1.0,
                                                                padding='post')
        y_dev = tf.keras.preprocessing.sequence.pad_sequences(dev_labels,
                                                              maxlen=max_len,
                                                              value=-1.0,
                                                              padding='post')

    else:
        y_train = np.array(train_labels).reshape(-1, 1)
        y_dev = np.array(dev_labels).reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=test_size,
                                                      random_state=seed)

    x_dev = tokenizer.texts_to_sequences(dev_sents)
    x_dev = tf.keras.preprocessing.sequence.pad_sequences(
        x_dev, maxlen=max_len)     # pad to the longest train sequence length

    return (x_train, x_val, y_train,
            y_val), (x_dev, y_dev), (max_len, len(tokenizer.word_index))