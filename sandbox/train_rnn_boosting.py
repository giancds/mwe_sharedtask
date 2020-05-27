import os

import tensorflow as tf

from sklearn.ensemble import AdaBoostClassifier

from classifiers import BoostedClassifier
from preprocess import Features, load_dataset, pre_process_data
from evaluation import evaluate
from utils import get_callbacks, get_optimizer, get_class_weights
from utils import define_rnn_flags, build_model_name

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
FLAGS = define_rnn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)

# define which feature we can use to train de model
_FEATURE = Features.upos

if FLAGS.feature == 'xpos':
    _FEATURE = Features.xpos

elif FLAGS.feature == 'deprel':
    _FEATURE = Features.deprel

model_name = build_model_name('sentlevel_boost', FLAGS)

# #####
# Loading data
#

print('Pre-processing data...')

# train dataset

# train_files = ['data/GA/train.cupt']
train_files = []
train_sents, train_labels = load_dataset(train_files, feature=_FEATURE)

# validation/dev dataset
# dev_files = ['data/GA/dev.cupt']
dev_files = []
dev_sents, dev_labels = load_dataset(dev_files, feature=_FEATURE, train=False)

train_data, dev_data, (max_len, n_tokens) = pre_process_data(
    (train_sents, train_labels), (dev_sents, dev_labels),
    seed=SEED)

x_train, x_val, y_train, y_val = train_data
x_dev, y_dev = dev_data

# need this for keras' loss functions
if FLAGS.output_size > 1:
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
            mask_zero=True,
            input_length=x_train.shape[1],
            embeddings_initializer=tf.random_uniform_initializer(
                minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED)))
    if FLAGS.spatial_dropout:
        model.add(tf.keras.layers.SpatialDropout1D(FLAGS.dropout))
    else:
        model.add(tf.keras.layers.Dropout(FLAGS.dropout))

    # LSTMs
    for layer in range(FLAGS.n_layers):
        return_sequences = False if layer == FLAGS.n_layers - 1 else True
        layer = tf.keras.layers.LSTM(
            FLAGS.lstm_size,
        #  dropout=FLAGS.lstm_dropout,
            recurrent_dropout=FLAGS.lstm_recurrent_dropout,
            return_sequences=return_sequences,
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
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-FLAGS.init_scale,
                    maxval=FLAGS.init_scale,
                    seed=SEED)))
    else:
        model.add(
            tf.keras.layers.Dense(
                2,
                activation=FLAGS.output_activation,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-FLAGS.init_scale,
                    maxval=FLAGS.init_scale,
                    seed=SEED)))

    # compiling model
    model.compile(loss=FLAGS.loss_function,
                  optimizer=get_optimizer(FLAGS),
                  metrics=['accuracy'])

    print(model.summary())

    return model


keras_model = BoostedClassifier(build_fn=build_model,
                                batch_size=FLAGS.batch_size,
                                epochs=FLAGS.max_epochs,
                                callbacks=get_callbacks(FLAGS, model_name),
                                verbose=FLAGS.verbose,
                                class_weight=get_class_weights(
                                    FLAGS.weighted_loss, train_labels),
                                validation_data=(x_val, y_val))

classifier = AdaBoostClassifier(base_estimator=keras_model,
                                n_estimators=FLAGS.n_estimators,
                                learning_rate=FLAGS.boost_lr,
                                random_state=SEED)

classifier.fit(x_train, y_train)

# #####
# Evaluation time
#
evaluate(classifier, (x_dev, y_dev), boosting=True)
