import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from preprocess import extract_dataset, Features
from evaluation import evaluate
from utils import get_callbacks, get_optimizer, get_class_weights
from utils import define_cnn_flags, build_model_name

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# #####
# Hyper-parametsr definitions
#

# pylint: disable=W0613,C0103,C0112
SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#
FLAGS = define_cnn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)

# define which feature we can use to train de model

model_name = build_model_name('sentlevel', FLAGS)

print('\nModel name {}\n'.format(model_name))

# #####
# Loading data
#

print('Pre-processing data...')

# train dataset

# train_files = ['data/GA/train.cupt']
train_files = []
for root, dirs, files in os.walk('data/'):
    for file in files:
        if file == 'train.cupt':
            train_files.append(os.path.join(root, file))

tmp = FLAGS.feature.split('+')
features = []
for f in tmp:
    if f == 'upos':
        features.append(Features.upos)

    elif f == 'deprel':
        features.append(Features.deprel)

train_dataset = [[]]
for i, feature in enumerate(features):
    tmp = extract_dataset(train_files, feature=feature)
    if i == 0:
        train_sents = [d[0] for d in tmp]
        train_labels = [d[1] for d in tmp]
        train_dataset.append(train_labels)
    else:
        train_sents = [d[0] for d in tmp]
    train_dataset[0].append(train_sents)

tokenizers = []
x_train = []
for dataset in train_dataset[0]:
    tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ', filters='')
    tokenizer.fit_on_texts(dataset)
    _x_train = tokenizer.texts_to_sequences(dataset)
    _x_train = tf.keras.preprocessing.sequence.pad_sequences(
        _x_train)     # pad to the longest sequence length
    tokenizers.append(tokenizer)
    x_train.append(np.expand_dims(np.array(_x_train), axis=2))

x_train = np.concatenate(x_train, axis=2)

max_len = x_train.shape[1]

y_train = np.array(train_labels).reshape(-1, 1)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  random_state=SEED)

# validation/dev dataset

dev_files = []
for root, dirs, files in os.walk('data/'):
    for file in files:
        if file == 'dev.cupt':
            dev_files.append(os.path.join(root, file))

dev_dataset = [[]]
for i, feature in enumerate(features):
    tmp = extract_dataset(train_files, feature=feature)
    if i == 0:
        dev_sents = [d[0] for d in tmp]
        dev_labels = [d[1] for d in tmp]
    else:
        dev_sents = [d[0] for d in tmp]

x_dev = []
for i, dataset in enumerate(dev_dataset):
    _x_dev = tokenizer.texts_to_sequences(dataset)
    _x_dev = tf.keras.preprocessing.sequence.pad_sequences(_x_dev,
                                                           maxlen=max_len)
    x_dev.append(np.expand_dims(np.array(_x_dev), axis=2))

x_dev = np.concatenate(x_dev, axis=2)
y_dev = np.array(dev_labels).reshape(-1, 1)

# #####
# Building and training the model
#

print("Building model...")

model = tf.keras.Sequential()
# embedding
model.add(
    tf.keras.layers.Embedding(
        max([len(t.word_index) for t in tokenizers]) + 1,
        FLAGS.embed_dim,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        input_length=max_len,
        mask_zero=True,
        embeddings_initializer=tf.random_uniform_initializer(
            minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED)))

shape = model.layers[0].output_shape

model.add(tf.keras.layers.Reshape((shape[1], shape[3], shape[2])))

if FLAGS.spatial_dropout:
    model.add(tf.keras.layers.SpatialDropout1D(FLAGS.dropout))
else:
    model.add(tf.keras.layers.Dropout(FLAGS.dropout))

for filters in FLAGS.filters:
    model.add(
        tf.keras.layers.Conv2D(
            filters,
            FLAGS.ngram,
            padding='valid',
            activation='relu',
        #    strides=1,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED)))

    model.add(tf.keras.layers.GlobalMaxPool2D())

model.add(
    tf.keras.layers.Dense(FLAGS.lstm_size,
                          activation='relu',
                          kernel_initializer=tf.random_uniform_initializer(
                              minval=-FLAGS.init_scale,
                              maxval=FLAGS.init_scale,
                              seed=SEED)))

if FLAGS.output_size == 1:
    model.add(
        tf.keras.layers.Dense(1,
                              activation='sigmoid',
                              kernel_initializer=tf.random_uniform_initializer(
                                  minval=-FLAGS.init_scale,
                                  maxval=FLAGS.init_scale,
                                  seed=SEED)))
else:
    model.add(
        tf.keras.layers.Dense(2,
                              activation=FLAGS.output_activation,
                              kernel_initializer=tf.random_uniform_initializer(
                                  minval=-FLAGS.init_scale,
                                  maxval=FLAGS.init_scale,
                                  seed=SEED)))
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)


# compiling model
model.compile(loss=FLAGS.loss_function,
              optimizer=get_optimizer(FLAGS),
              metrics=['accuracy'])

print(model.summary())


print('Train...')
model.fit(x_train,
          y_train,
          class_weight=get_class_weights(FLAGS.weighted_loss, train_labels),
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.max_epochs,
          callbacks=get_callbacks(FLAGS, model_name),
          verbose=FLAGS.verbose,
          validation_data=(x_val, y_val))

# #####
# Evaluation time
#
evaluate(model, test_data=(x_dev, y_dev))
