import os

import numpy as np
from tensorflow.compat.v1 import flags

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout
from keras.initializers import RandomUniform
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks.tensorboard_v1 import TensorBoard

from sklearn.metrics import confusion_matrix, classification_report

from preprocess import extract_dataset

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

upos = 18

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

flags.DEFINE_integer("embed_dim", 30, "Dimension of embbeddings.")

flags.DEFINE_boolean("spatial_dropout", False, "Whether or  to use spatial dropout for Embbeddings.")

flags.DEFINE_float("dropout", 0.1, "Embbeddings dropout.")

flags.DEFINE_boolean("bilstm", False,
                     "Whether or not to use bidirectional LSTMs")

flags.DEFINE_integer("lstm_size", 200, "Dimension of LSTM layers.")

flags.DEFINE_float("lstm_dropout", 0.2, "LSTM regular dropout.")

flags.DEFINE_float("lstm_recurrent_dropout", 0.0, "LSTM recurrent dropout.")

flags.DEFINE_integer("n_layers", 2, "Number of LSTM layers.")

flags.DEFINE_integer("batch_size", 32, "Size of batches.")

flags.DEFINE_string("optimizer", 'sgd',
                    "Which optimizer to use. One of adam, sgd and rmsprop.")

flags.DEFINE_float("learning_rate", 0.1, "Learning rate for the optimizer.")

flags.DEFINE_float("lr_decay", (1.0/1.15), "Rate to which we deca they learning rate during training.")

flags.DEFINE_integer("start_decay", 6, "Epoch to start the learning rate decay. To disable, set it to either 0 or to max_epochs")

flags.DEFINE_float("clipnorm", 5.0, "Max norm size to clipt the gradients.")

flags.DEFINE_float("init_scale", 0.05, "Range to initialize the weights of the model.")



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

train_dataset = extract_dataset(train_files)

train_sents = [d[0] for d in train_dataset]
train_labels = [d[1] for d in train_dataset]



tokenizer = Tokenizer(num_words=upos + 1, split=' ')  # +1 to account for padding later
tokenizer.fit_on_texts(train_sents)
x_train = tokenizer.texts_to_sequences(train_sents)
x_train = pad_sequences(x_train)     # pad to the longest sequence length

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

dev_dataset = extract_dataset(dev_files)

dev_sents = [d[0] for d in dev_dataset]
dev_labels = [d[1] for d in dev_dataset]

x_dev = tokenizer.texts_to_sequences(dev_sents)
x_dev = pad_sequences(
    x_dev,
    maxlen=x_train.shape[1])     # pad to the longest train sequence length

y_dev = np.array(dev_labels).reshape(-1, 1)

# #####
# Building and training the model
#

print("Building model...")

model = Sequential()
# embedding
model.add(Embedding(upos,
                    FLAGS.embed_dim,
                    input_length=x_train.shape[1],
                    embeddings_initializer=RandomUniform(minval=-FLAGS.init_scale,
                                                         maxval=FLAGS.init_scale,
                                                         seed=SEED)))
if FLAGS.spatial_dropout:
    model.add(SpatialDropout1D(FLAGS.dropout))
else:
    model.add(Dropout(FLAGS.dropout))

# LSTMs
for layer in range(FLAGS.n_layers):
    return_sequences = False if layer == FLAGS.n_layers - 1 else True
    layer = LSTM(FLAGS.lstm_size,
                #  dropout=FLAGS.lstm_dropout,
                 recurrent_dropout=FLAGS.lstm_recurrent_dropout,
                 return_sequences=return_sequences,
                 kernel_initializer=RandomUniform(minval=-FLAGS.init_scale,
                                                  maxval=FLAGS.init_scale,
                                                  seed=SEED),
                 recurrent_initializer=RandomUniform(minval=-FLAGS.init_scale,
                                                     maxval=FLAGS.init_scale,
                                                     seed=SEED),

                 )
    # if bidirectional
    if FLAGS.bilstm:
        layer = Bidirectional(layer)
    model.add(layer)
    model.add(Dropout(FLAGS.lstm_dropout))
# softmax
model.add(Dense(2, activation='softmax'))



optimizer = None
if str(FLAGS.optimizer).lower() == 'sgd':
    optimizer = SGD(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'adam':
    optimizer = Adam(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'rmsprop':
    optimizer = RMSprop(learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

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


def lr_scheduler(epoch):
    lr_decay = FLAGS.lr_decay **  max(epoch - FLAGS.start_decay, 0.0)
    return lt * lr_decay


if FLAGS.start_epoch > 0:
    lrate = LearningRateScheduler(lr_scheduler)
    callbacks.append(lrate)


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

y_pred = model.predict(x_dev).argmax(axis=1)

print('Confusion matrix:')
print(confusion_matrix(y_dev, y_pred))

print('\n\nReport')
print(classification_report(y_dev, y_pred))
