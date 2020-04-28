import os

import tensorflow as tf

from preprocess import Features, load_dataset, pre_process_data
from evaluation import evaluate
from utils import get_callbacks, get_optimizer, get_class_weights
from utils import define_cnn_flags, build_cnn_name

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# pylint: disable=W0613,C0103,C0112
SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#
FLAGS = define_cnn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)

# define which feature we can use to train de model

model_name = build_cnn_name('sentlevel', FLAGS)


# filters = FLAGS.filters
# filters = filters.replace('[', '').replace(']', '').split(',')
# filters = [int(i) for i in filters]
FLAGS.filters = [int(i) for i in FLAGS.filters]
# #####
# Loading data
#

print('Pre-processing data...')
tmp = FLAGS.feature.split('+')
features = []
for f in tmp:
    if f == 'upos':
        features.append(Features.upos)

    elif f == 'deprel':
        features.append(Features.deprel)

# train_files = ['data/GA/train.cupt']
train_files = []
train_sents, train_labels = load_dataset(train_files, features, cnn=True)

# validation/dev dataset
# dev_files = ['data/GA/dev.cupt']
dev_files = []
dev_sents, dev_labels = load_dataset(dev_files, features, cnn=True, train=False)

train_data, dev_data, (max_len, n_tokens) = pre_process_data(
    (train_sents, train_labels), (dev_sents, dev_labels), seed=SEED, cnn=True)

x_train, x_val, y_train, y_val = train_data
x_dev, y_dev = dev_data

# #####
# Building and training the model
#

print("Building model...")

model = tf.keras.Sequential()
# embedding
model.add(
    tf.keras.layers.Embedding(
        n_tokens + 1,
        FLAGS.embed_dim,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        input_length=max_len,
        mask_zero=True,
        embeddings_initializer=tf.random_uniform_initializer(
            minval=-FLAGS.init_scale, maxval=FLAGS.init_scale, seed=SEED)))

shape = model.layers[0].output_shape

model.add(tf.keras.layers.Reshape((shape[1], shape[3], shape[2])))

if FLAGS.spatial_dropout:
    model.add(tf.keras.layers.SpatialDropout2D(FLAGS.emb_dropout))
else:
    model.add(tf.keras.layers.Dropout(FLAGS.emb_dropout))

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
    model.add(tf.keras.layers.MaxPooling2D())

if FLAGS.global_pooling == 'average':
    model.add(tf.keras.layers.GlobalAveragePooling2D())
else:
    model.add(tf.keras.layers.GlobalMaxPool2D())

for layer in range(FLAGS.n_layers):
    model.add(
        tf.keras.layers.Dense(FLAGS.dense_size,
                            activation='relu',
                            kernel_initializer=tf.random_uniform_initializer(
                                minval=-FLAGS.init_scale,
                                maxval=FLAGS.init_scale,
                                seed=SEED)))
    model.add(tf.keras.layers.Dropout(FLAGS.dropout))

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
