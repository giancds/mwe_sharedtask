import os

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from preprocess import Features, load_dataset, pre_process_data
from evaluation import evaluate, MetricsReporterCallback
from utils import get_callbacks, get_optimizer
from utils import define_rnn_flags, build_model_name, convert_flags_to_dict

import ray
from ray import tune
from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import Repeater
from hyperopt import hp

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

# model_name = build_model_name('sentlevel', FLAGS)

_config = convert_flags_to_dict(FLAGS)
_config["is_dev"] = False

cwd = os.getcwd()

# #####
# Loading data
#

print('Pre-processing data...')

# train dataset

# train_files = ['data/GA/train.cupt']
train_files = [cwd + '/data/GA/train.cupt'] if _config["is_dev"] else []
train_sents, train_labels = load_dataset(train_files, feature=_FEATURE)

# validation/dev dataset
dev_files = [cwd + '/data/GA/dev.cupt'] if _config["is_dev"] else []
dev_sents, dev_labels = load_dataset(dev_files, feature=_FEATURE, train=False)

train_data, dev_data, (max_len, n_tokens) = pre_process_data(
    (train_sents, train_labels), (dev_sents, dev_labels), seed=SEED)

_x_train, _x_val, _y_train, _y_val = train_data
_x_dev, _y_dev = dev_data

_config["x_train"] = _x_train
_config["x_val"] = _x_val
_config["y_train"] = _y_train
_config["y_val"] = _y_val

_config["x_dev"] = _x_dev
_config["y_dev"] = _y_dev

_config["n_tokens"] = n_tokens
_config["max_len"] = max_len

# #####
# Building and training the model
#

print("Building model...")

def train_model(config):

    model_name = build_model_name('sentlevel', config)

    model = tf.keras.Sequential()
    # embedding
    model.add(
        tf.keras.layers.Embedding(
            config["n_tokens"] + 1,
            config["embed_dim"],
            input_length=config["max_len"],
            mask_zero=True,
            embeddings_initializer=tf.random_uniform_initializer(
                minval=-config["init_scale"],
                maxval=config["init_scale"],
                seed=SEED)))
    if config["spatial_dropout"]:
        model.add(tf.keras.layers.SpatialDropout1D(config["dropout"]))
    else:
        model.add(tf.keras.layers.Dropout(config["dropout"]))

    # LSTMs
    for layer in range(config["n_layers"]):
        return_sequences = False if layer == config["n_layers"] - 1 else True
        layer = tf.keras.layers.LSTM(
            config["lstm_size"],
        #  dropout=FLAGS.lstm_dropout,
            recurrent_dropout=config["lstm_recurrent_dropout"],
            return_sequences=return_sequences,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-config["init_scale"],
                maxval=config["init_scale"],
                seed=SEED),
            recurrent_initializer=tf.random_uniform_initializer(
                minval=-config["init_scale"],
                maxval=config["init_scale"],
                seed=SEED),
        )
        # if bidirectional
        if config["bilstm"]:
            layer = tf.keras.layers.Bidirectional(layer)
        model.add(layer)
        model.add(tf.keras.layers.Dropout(config["lstm_dropout"]))

    if config["output_size"] == 1:
        model.add(
            tf.keras.layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-config["init_scale"],
                    maxval=config["init_scale"],
                    seed=SEED)))
        y_train = config["y_train"]
        y_val = config["y_val"]
    else:
        model.add(
            tf.keras.layers.Dense(
                2,
                activation=config["output_activation"],
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-config["init_scale"],
                    maxval=config["init_scale"],
                    seed=SEED)))
        y_train = tf.keras.utils.to_categorical(config["y_train"])
        y_val = tf.keras.utils.to_categorical(config["y_val"])

    if config["optimizer"] == 'adam':
        optimizer = tf.keras.optimizers.Adam
    elif config["optimizer"] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop
    else:
        optimizer = tf.keras.optimizers.SGD

    # compiling model
    model.compile(loss=config["loss_function"],
                  optimizer=optimizer(learning_rate=config["learning_rate"],
                                      clipnorm=config["clipnorm"]),
                  metrics=['accuracy'])

    print(model.summary())

    class_weights = None
    if config["weighted_loss"]:
        weights = class_weight.compute_class_weight(
            'balanced', np.array([0, 1]), np.array([i for i in train_labels]))
        class_weights = {}

        for i in range(weights.shape[0]):
            class_weights[i] = weights[i]

    print('Class weights: {}'.format(class_weights))


    checkpoint = tf.keras.callbacks.ModelCheckpoint(config["train_dir"] +
                                                    model_name,
                                                    save_best_only=True)
    callbacks = [
        MetricsReporterCallback(
            custom_validation_data=(config["x_val"], y_val)),
        checkpoint]

    if config["early_stop_patience"] > 0:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=config["early_stop_delta"],
            patience=config["early_stop_patience"])
        callbacks.append(early_stop)

    if config["log_tensorboard"]:
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=config["train_dir"] + '/logs')
        callbacks.append(tensorboard)

    def lr_scheduler(epoch, lr):     # pylint: disable=C0103
        lr_decay = config["lr_decay"]**max(epoch - config["start_decay"], 0.0)
        return lr * lr_decay

    if config["start_decay"] > 0:
        lrate = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        callbacks.append(lrate)

    print('Train...')
    model.fit(config["x_train"],
              y_train,
              class_weight=class_weights,
              batch_size=config["batch_size"],
              epochs=config["max_epochs"],
              callbacks=callbacks,
              verbose=2,
              validation_data=(config["x_val"], y_val))

    # #####
    # Evaluation time
    #
    evaluate(model, test_data=(config["x_dev"], config["y_dev"]))


search_space = {
    "embed_dim":
        hp.choice("embed_dim", [10, 20, 30, 50, 75, 100]),
    "emb_dropout":
        hp.uniform("emb_dropout", 0.0, 0.9),
    "dropout":
        hp.uniform("dropout", 0.0, 0.9),
    "lstm_dropout":
        hp.uniform("lstm_dropout", 0.0, 0.9),
    "lstm_recurrent_dropout":
        hp.uniform("lstm_recurrent_dropout", 0.0, 0.9),
    "bilstm":
        hp.choice("bilstm", [True, False]),
    "spatial_dropout":
        hp.choice("spatial_dropout", [True, False]),
    "init_scale":
        hp.loguniform("init_scale", np.log(1e-2), np.log(1e-1)),
    "n_layers":
        hp.choice("n_layers", [1, 2, 3, 4]),
    "lstm_size":
        hp.choice("lstm_size", [50, 100, 150, 200, 250, 500]),
    "max_epochs":
        hp.choice("max_epochs", [30, 50, 70]),
    "early_stop_delta":
        hp.choice("early_stop_delta", [0.001, 0.0001]),
    "early_stop_patience":
        hp.choice("early_stop_patience", [10, 20]),
    "output_activation":
        hp.choice("output_activation", ['sigmoid', 'softmax']),
    "feature":
        hp.choice("feature", ["upos", "deprel"]),
    "clipnorm":
        hp.choice("clipnorm", [0.5, 1.0, 2.5, 5.0, 10.0]),
    "learning_rate":
        hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-0)),
    "optimizer":
        hp.choice("optimizer", ['adam', 'rmsprop']),
    "batch_size":
        hp.choice("batch_size", [20, 24, 32, 64, 128]),
}


_config.update({
    "threads": 1,
    "output_size": 2,
    "start_decay": 0
})

results = tune.run_experiments(
    tune.Experiment(
        run=train_model,
        name="tune-rnn",
        config=_config,
         stop={
            "keras_info/label1_f1_score": 0.9,
            "training_iteration": 10**8
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        num_samples=20,
        checkpoint_freq=0,
        checkpoint_at_end=False),
    scheduler=AsyncHyperBandScheduler(
        time_attr="epoch",
        metric="keras_info/label1_f1_score",
        mode="max",
        max_t=400,
        grace_period=20),
    search_alg=HyperOptSearch(
            search_space,
            metric="keras_info/label1_f1_score",
            mode="max",
            random_state_seed=SEED,
            points_to_evaluate=[{
                "embed_dim": 2,
                "emb_dropout": 0.1,
                "dropout": 0.1,
                "lstm_dropout": 0.2,
                "lstm_recurrent_dropout": 0.0,
                "bilstm": 1,
                "spatial_dropout": 0,
                "init_scale": 0.05,
                "n_layers": 0,
                "lstm_size": 1,
                "max_epochs": 1,
                "early_stop_delta": 0,
                "early_stop_patience": 0,
                "optimizer": 1,
                "output_activation": 0,
                "feature":  0,
                "clipnorm": 1,
                "learning_rate": 0.0001,
                "optimizer": 0,
                "batch_size": 2,
            }]),
    verbose=1,)


results.dataframe().to_csv(_config["train_dir"] + '/rnn_results.csv')