import os

import numpy as np
from tensorflow.compat.v1 import flags

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM, Bidirectional
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks.tensorboard_v1 import TensorBoard

from sklearn.metrics import confusion_matrix, classification_report

from preprocess import extract_dataset

# #####
# Hyper-parametsr definitions
#

# pylint: disable=W0613,C0103,C0112
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#

upos = 18     # number of upos in the train dataset
n_labels = 2

flags.DEFINE_integer("max_epochs", 1,
                     "Max number of epochs to train the models")

flags.DEFINE_integer("early_stop_patience", 10,
                     "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_float("early_stop_delta", 0.001,
                   "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_boolean("log_tensorboard", False,
                     "Whether or not to log info using tensorboard")

flags.DEFINE_string("model_name", "pwmodel.ckpt", "Model name")

flags.DEFINE_string("train_dir",
                    os.path.join(BASE_DIR, TRAIN_DIR) + "/", "Train directory")

flags.DEFINE_integer("embed_dim", 100, "Dimension of embbeddings.")

flags.DEFINE_float("spatial_dropout", 0.4, "Embbeddings dropout.")

flags.DEFINE_boolean("bilstm", False,
                     "Whether or not to use bidirectional LSTMs")

flags.DEFINE_integer("lstm_size", 100, "Dimension of LSTM layers.")

flags.DEFINE_float("lstm_dropout", 0.2, "LSTM regular dropout.")

flags.DEFINE_float("lstm_recurrent_dropout", 0.2, "LSTM recurrent dropout.")

flags.DEFINE_integer("n_layers", 1, "Number of LSTM layers.")

flags.DEFINE_integer("batch_size", 32, "Size of batches.")

flags.DEFINE_string("optimizer", 'adam',
                    "Which optimizer to use. One of adam, sgd and rmsprop.")

flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for the optimizer.")

flags.DEFINE_float("clipnorm", 0.1, "Max norm size to clipt the gradients.")

FLAGS = flags.FLAGS

# #####
# Loading data
#

print('Pre-processing data...')

# train dataset

train_files = ['data/PT/train.cupt']
# train_files = []
# for root, dirs, files in os.walk('data/'):
#     for file in files:
#         if file == 'train.cupt':
#             train_files.append(os.path.join(root, file))

train_dataset = extract_dataset(train_files, per_word=True)

train_sents = [d[0] for d in train_dataset]
train_labels = [d[1] for d in train_dataset]

tokenizer_pos = Tokenizer(num_words=upos + 1,
                          split=' ')     # +1 to account for padding later

tokenizer_pos.fit_on_texts(train_sents)
x_train = tokenizer_pos.texts_to_sequences(train_sents)
x_train = pad_sequences(x_train, value=0.0,
                        padding='post')     # pad to the longest sequence length

# tokenizer_lab.fit_on_texts(train_labels)
# y_train = tokenizer_lab.texts_to_sequences(train_labels)
y_train = pad_sequences(train_labels,
                        maxlen=x_train.shape[1],
                        value=-1.0,
                        padding='post')     # pad to the longest sequence length

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  random_state=42)

dev_files = ['data/PT/dev.cupt']
# dev_files = []
# for root, dirs, files in os.walk('data/'):
#     for file in files:
#         if file == 'dev.cupt':
#             dev_files.append(os.path.join(root, file))

dev_dataset = extract_dataset(dev_files, per_word=True)

dev_sents = [d[0] for d in dev_dataset]
dev_labels = [d[1] for d in dev_dataset]

x_dev = tokenizer_pos.texts_to_sequences(dev_sents)
x_dev = pad_sequences(x_dev, maxlen=x_train.shape[1], value=0.0, padding='post')

# y_dev = tokenizer_lab.texts_to_sequences(dev_labels)
y_dev = pad_sequences(dev_labels,
                      maxlen=x_train.shape[1],
                      value=-1.0,
                      padding='post')

# #####
# Building and training the model
#

print("Building model...")

model = Sequential()
# embedding
model.add(Embedding(upos, FLAGS.embed_dim, input_length=x_train.shape[1]))
model.add(SpatialDropout1D(FLAGS.spatial_dropout))
# LSTMs
for layer in range(FLAGS.n_layers):
    # return_sequences = False if layer == FLAGS.n_layers - 1 else True
    return_sequences = True
    layer = LSTM(FLAGS.lstm_size,
                 dropout=FLAGS.lstm_dropout,
                 recurrent_dropout=FLAGS.lstm_recurrent_dropout,
                 return_sequences=return_sequences)
    # if bidirectional
    if FLAGS.bilstm:
        layer = Bidirectional(layer)
    model.add(layer)
# softmax
model.add(Dense(n_labels, activation='softmax'))

optimizer = None
if str(FLAGS.optimizer).lower() == 'sgd':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'adam':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'rmsprop':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

# compiling model
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())
checkpoint = ModelCheckpoint(FLAGS.train_dir + FLAGS.model_name,
                             save_best_only=True)
callbacks = [checkpoint]

if FLAGS.early_stop_patience > 0:
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=FLAGS.early_stop_delta,
                               patience=FLAGS.early_stop_patience)
    callbacks.append(early_stop)

if FLAGS.log_tensorboard:
    tensorboard = TensorBoard(log_dir=FLAGS.train_dir + '/logs')
    callbacks.append(tensorboard)

print('Train...')
model.fit(x_train,
          np_utils.to_categorical(y_train),
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.max_epochs,
          callbacks=callbacks,
          validation_data=(x_val, np_utils.to_categorical(y_val)))

# #####
# Evaluation time
#

_y_pred = model.predict(x_dev).argmax(axis=2)


lens = [len(i for i in dev_labels]
y_true = []
y_pred = []
for i in range(y_dev.shape[0]):
    # y_true += y_dev[i, -lens[i]:].tolist()
    # y_pred += _y_pred[i, -lens[i]:].tolist()
    y_true += y_dev[i, 0:lens[i]].tolist()
    y_pred += _y_pred[i, 0:lens[i]].tolist()

print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))

print('\n\nReport')
print(classification_report(y_true, y_pred))
