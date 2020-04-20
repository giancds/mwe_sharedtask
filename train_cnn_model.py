import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from preprocess import extract_dataset, build_model_name, Features
from evaluation import evaluate

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
upos = 18
flags = tf.compat.v1.flags

flags.DEFINE_integer("max_epochs", 100,
                     "Max number of epochs to train the models")

flags.DEFINE_integer("early_stop_patience", 10,
                     "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_float("early_stop_delta", 0.001,
                   "How many training steps to monitor. Set to 0 to ignore.")

flags.DEFINE_boolean("log_tensorboard", False,
                     "Whether or not to log info using tensorboard")

flags.DEFINE_string("train_dir",
                    os.path.join(BASE_DIR, TRAIN_DIR) + "/", "Train directory")

flags.DEFINE_integer("embed_dim", 20, "Dimension of embbeddings.")

flags.DEFINE_list("filters", [128], "Dimension of embbeddings.")

flags.DEFINE_integer("ngram", 3, "Dimension of embbeddings.")

flags.DEFINE_boolean("spatial_dropout", False,
                     "Whether or  to use spatial dropout for Embbeddings.")

flags.DEFINE_float("dropout", 0.1, "Embbeddings dropout.")

flags.DEFINE_boolean("bilstm", False,
                     "Whether or not to use bidirectional LSTMs")

flags.DEFINE_integer("lstm_size", 50, "Dimension of LSTM layers.")

flags.DEFINE_float("lstm_dropout", 0.0, "LSTM regular dropout.")

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

flags.DEFINE_integer("batch_size", 32, "Size of batches.")

flags.DEFINE_string("optimizer", 'sgd',
                    "Which optimizer to use. One of adam, sgd and rmsprop.")

flags.DEFINE_float("learning_rate", 1.0, "Learning rate for the optimizer.")

flags.DEFINE_float("lr_decay", (1.0 / 1.15),
                   "Rate to which we deca they learning rate during training.")

flags.DEFINE_integer(
    "start_decay", 6,
    "Epoch to start the learning rate decay. To disable, set it to either 0 or to max_epochs"
)

flags.DEFINE_float("clipnorm", 5.0, "Max norm size to clipt the gradients.")

flags.DEFINE_float("init_scale", 0.05,
                   "Range to initialize the weights of the model.")

flags.DEFINE_integer("verbose", 1, "Verbosity of training")

flags.DEFINE_string("feature", 'upos+xpos+deprel',
                    "Which feature to use when training de model.")
FLAGS = flags.FLAGS

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
model.compile(loss=FLAGS.loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

checkpoint = tf.keras.callbacks.ModelCheckpoint(FLAGS.train_dir + model_name,
                                                save_best_only=True)
callbacks = [checkpoint]

if FLAGS.early_stop_patience > 0:
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
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
        'balanced', np.array([0, 1]), np.array([i for i in train_labels]))
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
evaluate(model, test_data=(x_dev, y_dev))
