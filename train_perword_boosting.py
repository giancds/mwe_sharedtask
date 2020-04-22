import os

import numpy as np
import tensorflow as tf

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from classifiers import BoostedTemporalClassifier
from preprocess import extract_dataset, Features, load_dataset, pre_process_data
from evaluation import evaluate
from utils import get_callbacks, get_optimizer, get_class_weights
from utils import define_rnn_flags, build_model_name


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# #####
# Hyper-parameters definitions
#

# pylint: disable=W0613,C0103,C0112
SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#

FLAGS = define_rnn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)


# define which feature we can use to train de model
_FEATURE = Features.upos

if FLAGS.feature == 'xpos':
    _FEATURE = Features.xpos

elif FLAGS.feature == 'deprel':
    _FEATURE = Features.deprel

model_name = build_model_name('perword', FLAGS)

# #####
# Loading data
#

print('Pre-processing data...')

# train dataset

# train_files = ['data/GA/train.cupt']
train_files = []
train_sents, train_labels = load_dataset(train_files, per_word=True, feature=_FEATURE)


# test dataset
# dev_files = ['data/GA/dev.cupt']
dev_files = []
dev_sents, dev_labels = load_dataset(dev_files, per_word=True, feature=_FEATURE)

dev_files = []
dev_sents, dev_labels = load_dataset(dev_files, per_word=True, feature=_FEATURE, train=False)

train_data, dev_data, (max_len, n_tokens) = pre_process_data(
    (train_sents, train_labels), (dev_sents, dev_labels),
    seed=SEED,
    pad_targets=True)

x_train, x_val, y_train, y_val = train_data
x_dev, y_dev = dev_data

# need this for keras' loss functions
y_val = tf.keras.utils.to_categorical(y_val)

# #####
# Building and training the model
#

print("Building model...")

def build_model():
    model = tf.keras.Sequential()
    # embedding
    model.add(
        tf.keras.layers.Embedding(
            n_tokens + 1,
            FLAGS.embed_dim,
            input_length=x_train.shape[1],
            mask_zero=True,
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

        model.add(
            tf.keras.layers.Dense(2,
                                activation=FLAGS.output_activation,
                                kernel_initializer=tf.random_uniform_initializer(
                                    minval=-FLAGS.init_scale,
                                    maxval=FLAGS.init_scale,
                                    seed=SEED)))

    # compiling model
    model.compile(loss='binary_crossentropy',
                optimizer=get_optimizer(FLAGS),
                sample_weight_mode='temporal',
                metrics=['accuracy'])

    print(model.summary())

    return model

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

keras_model = BoostedTemporalClassifier(
    build_fn=build_model,
    batch_size=FLAGS.batch_size,
    epochs=FLAGS.max_epochs,
    callbacks=get_callbacks(FLAGS, model_name),
    verbose=FLAGS.verbose,
    validation_data=(x_val, y_val))

classifier = AdaBoostClassifier(base_estimator=keras_model,
                                n_estimators=FLAGS.n_estimators,
                                learning_rate=FLAGS.boost_lr,
                                random_state=SEED)

classifier.fit(x_train.flatten(), y_train.flatten())
# classifier.fit(x_train, y_train)


# #####
# Evaluation time
#
evaluate(classifier, (x_dev, y_dev), perword=True, boosting=True, seq_lens=[len(i) for i in dev_labels])