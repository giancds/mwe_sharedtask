import os

import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum
from sklearn.model_selection import train_test_split


class Features(Enum):
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


def load_dataset(train_files, feature, per_word=False):

    if len(train_files) == 0:
        _train_files = []
        for root, _, files in os.walk('data/'):
            for file in files:
                if file == 'train.cupt':
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
