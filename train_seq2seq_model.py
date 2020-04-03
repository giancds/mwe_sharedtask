import os

import numpy as np
from tensorflow.compat.v1 import flags

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Input, Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks.tensorboard_v1 import TensorBoard

from sklearn.metrics import confusion_matrix, classification_report

from preprocess import extract_dataset

# #####
# Hyper-parametsr definitions
#

# pylint: disable=W0613,C0103,C0112
seed = 1701
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#

EOS = '_EOS '
EOL = ' _EOL'

upos = 18
n_labels = len(['0', '1']) + len([EOS, EOL])

flags.DEFINE_integer("max_epochs", 100,
                     "Max number of epochs to train the models")

flags.DEFINE_integer("early_stop_patience", 10,
                     "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_float("early_stop_delta", 0.001,
                   "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_boolean("log_tensorboard", False,
                     "Whether or not to log info using tensorboard")

flags.DEFINE_string("model_name", "model.ckpt", "Model name")

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

upos = 18     # number of upos in the train dataset

# train dataset

train_files = []
for root, dirs, files in os.walk('data/'):
    for file in files:
        if file == 'train.cupt':
            train_files.append(os.path.join(root, file))

train_dataset = extract_dataset(train_files, per_word=True)

train_sents = [d[0] for d in train_dataset]
train_labels = [(EOS + d[1] + EOL).strip() for d in train_dataset]

tokenizer_pos = Tokenizer(num_words=upos, split=' ')
tokenizer_lab = Tokenizer(num_words=n_labels, split=' ')

tokenizer_pos.fit_on_texts(train_sents)
x_train = tokenizer_pos.texts_to_sequences(train_sents)
x_train = pad_sequences(x_train)     # pad to the longest sequence length

tokenizer_lab.fit_on_texts(train_labels)
y_train = tokenizer_lab.texts_to_sequences(train_labels)
y_train = pad_sequences(y_train, maxlen=x_train.shape[1])     # pad to the longest sequence length

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  random_state=42)

# validation/dev dataset

dev_files = []
for root, dirs, files in os.walk('data/'):
    for file in files:
        if file == 'dev.cupt':
            dev_files.append(os.path.join(root, file))

dev_dataset = extract_dataset(dev_files, per_word=True)

dev_sents = [d[0] for d in dev_dataset]
dev_labels = [(EOS + d[1] + EOL).strip() for d in dev_dataset]

x_dev = tokenizer_pos.texts_to_sequences(dev_sents)
x_dev = pad_sequences(x_dev, maxlen=x_train.shape[1])

y_dev = tokenizer_lab.texts_to_sequences(dev_labels)
y_dev = pad_sequences(y_dev, maxlen=x_train.shape[1])

# #####
# Building and training the model
#

print("Building model...")

# encoder side
encoder_inputs = Input(shape=(None,))
x = Embedding(upos, FLAGS.lstm_size)(encoder_inputs)
x, state_h, state_c = LSTM(FLAGS.lstm_size, return_state=True)(x)
encoder_states = [state_h, state_c]

# decoder side
decoder_inputs = Input(shape=(None,))
x = Embedding(n_labels, FLAGS.lstm_size)(decoder_inputs)
x = LSTM(FLAGS.lstm_size, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(n_labels, activation='softmax')(x)

# model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# optimizers
optimizer = None
if str(FLAGS.optimizer).lower() == 'sgd':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'adam':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'rmsprop':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

# compiling model
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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
model.fit([x_train, y_train[:, 0:-1]], np_utils.to_categorical((y_train[:, 1:])),
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.max_epochs,
          callbacks=callbacks,
          validation_data=([x_val, y_val[:, 0:-1]], np_utils.to_categorical((y_val[:, 1:])))

# #####
# Evaluation time
#


