import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from preprocess import extract_dataset, build_model_name, Features

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
verbose = 2
upos = 18     # number of upos in the train dataset
n_labels = 2

flags = tf.compat.v1.flags

# TODO: select features in all scripts

flags.DEFINE_integer("max_epochs", 1,
                     "Max number of epochs to train the models")

flags.DEFINE_integer("early_stop_patience", 10,
                     "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_float("early_stop_delta", 0.001,
                   "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_boolean("log_tensorboard", False,
                     "Whether or not to log info using tensorboard")

flags.DEFINE_string("train_dir",
                    os.path.join(BASE_DIR, TRAIN_DIR) + "/", "Train directory")

flags.DEFINE_integer("embed_dim", 10, "Dimension of embbeddings.")

flags.DEFINE_boolean("spatial_dropout", False,
                     "Whether or  to use spatial dropout for Embbeddings.")

flags.DEFINE_float("dropout", 0.2, "Embbeddings dropout.")

flags.DEFINE_boolean("bilstm", False,
                     "Whether or not to use bidirectional LSTMs")

flags.DEFINE_integer("lstm_size", 50, "Dimension of LSTM layers.")

flags.DEFINE_float("lstm_dropout", 0.5, "LSTM regular dropout.")

flags.DEFINE_float("lstm_recurrent_dropout", 0.0, "LSTM recurrent dropout.")

flags.DEFINE_integer("n_layers", 1, "Number of LSTM layers.")

flags.DEFINE_string("output_activation", 'sigmoid',
                    "Activation for the output layer.")

flags.DEFINE_integer(
    "output_size", 2,
    "Size of the output layer. Only relevant when using sigmoid output.")

flags.DEFINE_float(
    "output_threshold", 0.5,
    "Threshold to classify a sentence as idiomatic or not. Only relevant when using sigmoid output."
)

flags.DEFINE_string("loss_function", 'binary_crossentropy',
                    "Loss function to use during training.")

flags.DEFINE_boolean("weighted_loss", True,
                     "Whether or to use weighted loss for learning.")

flags.DEFINE_integer("batch_size", 24, "Size of batches.")

flags.DEFINE_string("optimizer", 'adam',
                    "Which optimizer to use. One of adam, sgd and rmsprop.")

flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for the optimizer.")

flags.DEFINE_float("lr_decay", 0.8,
                   "Rate to which we deca they learning rate during training.")

flags.DEFINE_integer(
    "start_decay", 0,
    "Epoch to start the learning rate decay. To disable, set it to either 0 or to max_epochs"
)

flags.DEFINE_float("clipnorm", 5.0, "Max norm size to clipt the gradients.")

flags.DEFINE_float("init_scale", 0.05,
                   "Range to initialize the weights of the model.")

flags.DEFINE_integer(
    "verbose", 2, "Verbosity of training"
)

flags.DEFINE_string("feature", 'upos',
                    "Which feature to use when training de model.")
FLAGS = flags.FLAGS


# define which feature we can use to train de model
_FEATURE = Features.upos

if FLAGS.feature == 'xpos':
    _FEATURE = Features.xpos

elif FLAGS.feature == 'deprel':
    _FEATURE = Features.deprel

#
#
model_name = build_model_name('perword', FLAGS)

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

train_dataset = extract_dataset(train_files, per_word=True, feature=_FEATURE)

train_sents = [d[0] for d in train_dataset]
train_labels = [d[1] for d in train_dataset]

tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ')

tokenizer.fit_on_texts(train_sents)
x_train = tokenizer.texts_to_sequences(train_sents)
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, value=0.0, padding='post')     # pad to the longest sequence length

y_train = tf.keras.preprocessing.sequence.pad_sequences(
    train_labels, maxlen=x_train.shape[1], value=-1.0,
    padding='post')     # pad to the longest sequence length

# traind and validation split
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  random_state=42)

# test dataset
# dev_files = ['data/GA/dev.cupt']
dev_files = []
for root, dirs, files in os.walk('data/'):
    for file in files:
        if file == 'dev.cupt':
            dev_files.append(os.path.join(root, file))

dev_dataset = extract_dataset(dev_files, per_word=True, feature=_FEATURE)

dev_sents = [d[0] for d in dev_dataset]
dev_labels = [d[1] for d in dev_dataset]

x_dev = tokenizer.texts_to_sequences(dev_sents)
x_dev = tf.keras.preprocessing.sequence.pad_sequences(x_dev,
                                                      maxlen=x_train.shape[1],
                                                      value=0.0,
                                                      padding='post')

y_dev = tf.keras.preprocessing.sequence.pad_sequences(dev_labels,
                                                      maxlen=x_train.shape[1],
                                                      value=-1.0,
                                                      padding='post')

# #####
# Building and training the model
#

print("Building model...")

model = tf.keras.Sequential()
# embedding
model.add(
    tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1,
        FLAGS.embed_dim,
        input_length=x_train.shape[1],
        # mask_zero=True,
        embeddings_initializer=tf.random_uniform_initializer(
            minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED)))
if FLAGS.spatial_dropout:
    model.add(tf.keras.layers.SpatialDropout1D(FLAGS.dropout))
else:
    model.add(tf.keras.layers.Dropout(FLAGS.dropout))

# LSTMs
for layer in range(FLAGS.n_layers):
    # return_sequences = False if layer == FLAGS.n_layers - 1 else True
    layer = tf.keras.layers.LSTM(
        FLAGS.lstm_size,
     #  dropout=FLAGS.lstm_dropout,
        recurrent_dropout=FLAGS.lstm_recurrent_dropout,
        return_sequences=True,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED),
        recurrent_initializer=tf.random_uniform_initializer(
            minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED),
    )
    # if bidirectional
    if FLAGS.bilstm:
        layer = tf.keras.layers.Bidirectional(layer)
    model.add(layer)
    model.add(tf.keras.layers.Dropout(FLAGS.lstm_dropout))

if FLAGS.output_size == 1:
    model.add(
        tf.keras.layers.Dense(1,
                              activation='sigmoid',
                              kernel_initializer=tf.random_uniform_initializer(
                                  minval=-FLAGS.init_scale,
                                  maxval=FLAGS.init_scale,
                                  seed=SEED)))
    y_train = np.expand_dims(y_train, axis=2)
    y_val = np.expand_dims(y_val, axis=2)
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

#
# optimizers
if str(FLAGS.optimizer).lower() == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate,
                                        clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                         clipnorm=FLAGS.clipnorm)

elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate,
                                            clipnorm=FLAGS.clipnorm)

# compiling model
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())
checkpoint = tf.keras.callbacks.ModelCheckpoint(FLAGS.train_dir + model_name,
                                                save_best_only=True)
callbacks = [checkpoint]

if FLAGS.early_stop_patience > 0:
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=FLAGS.early_stop_delta,
        patience=FLAGS.early_stop_patience)
    callbacks.append(early_stop)

if FLAGS.log_tensorboard:
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.train_dir +
                                                 '/logs')
    callbacks.append(tensorboard)


def lr_scheduler(epoch, lr):
    lr_decay = FLAGS.lr_decay**max(epoch - FLAGS.start_decay, 0.0)
    return lr * lr_decay


if FLAGS.start_decay > 0:
    lrate = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    callbacks.append(lrate)

# calculate class weights for the imbalanced case
class_weights = None
if FLAGS.weighted_loss:
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train.flatten()), y_train.flatten())
    weights = class_weight.compute_class_weight(
        'balanced', np.array([0, 1]),
        np.array([i for j in train_labels for i in j]))
    class_weights = {}
    for i in range(weights.shape[0]):
        class_weights[i] = weights[i]

print('Class weights: {}'.format(class_weights))

print('Train...')
model.fit(x_train,
          y_train,
          class_weight=class_weights,
          batch_size=FLAGS.batch_size,
          epochs=FLAGS.max_epochs,
          callbacks=callbacks,
          verbose=FLAGS.verbose,
          validation_data=(x_val, y_val))

# #####
# Evaluation time
#

if FLAGS.output_size == 1:
    _y_pred = model.predict(x_dev).astype('int')
else:
    _y_pred = model.predict(x_dev).argmax(axis=2)

lens = [len(i) for i in dev_labels]
y_pred = []
y_true = []
for i, l in enumerate(lens):
    y_pred += _y_pred[i, 0:l].tolist()
    y_true += y_dev[i, 0:l].tolist()

print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))

print('\n\nReport')
print(classification_report(y_true, y_pred))