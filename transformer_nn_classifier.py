#pylint: disable=invalid-name
import os
import pickle
import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from evaluation import MetricsReporterCallback, evaluate
from utils import build_model_name, convert_flags_to_dict, define_nn_flags

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "ray_results"

FLAGS = define_nn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)
FLAGS.layers = [int(i) for i in FLAGS.layers]

_config = convert_flags_to_dict(FLAGS)
_config["codes"] = (['DE', 'GA', 'HI', 'PT', 'ZH']
                    if FLAGS.language_code is 'all' else [FLAGS.language_code])

cwd = os.getcwd()


def train_model(config):

    model_name = build_model_name(config)

    with open('{}/data/{}.embdata.pkl'.format(cwd, config["bert_type"]),
              'rb') as f:
        data = pickle.load(f)

    x_train = np.concatenate(
        [data[code]['x_train'] for code in _config["codes"]], axis=0)
    y_train = np.concatenate(
        [data[code]['y_train'] for code in _config["codes"]], axis=0)
    print(x_train.shape, y_train.shape)

    x_dev = np.concatenate([data[code]['x_dev'] for code in _config["codes"]],
                           axis=0)
    y_dev = np.concatenate([data[code]['y_dev'] for code in _config["codes"]],
                           axis=0)
    print(x_dev.shape, y_dev.shape)

    del data

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.15,
                                                      random_state=SEED)

    model = tf.keras.Sequential()
    # embedding

    layer_config = ([config["layer_size"]] *
                    config["nlayers"] if config["nlayers"] > 0 and
                    config["layer_size"] > 0 else config["layers"])

    # Dense layers
    for i, layer_size in enumerate(layer_config):
        if i == 0:
            dense_layer = tf.keras.layers.Dense(
                layer_size,
                input_shape=(x_train.shape[-1],),
                activation=config["hidden_activation"])
        else:
            dense_layer = tf.keras.layers.Dense(
                layer_size, activation=config["hidden_activation"])
        model.add(dense_layer)
        model.add(tf.keras.layers.Dropout(config["dropout"]))

    if config["output_size"] == 1:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(
            tf.keras.layers.Dense(
                2,
                activation=config["output_activation"],
            ))

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
        weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(y_train),
                                                    y_train.reshape(-1))
        class_weights = {}

        for i in range(weights.shape[0]):
            class_weights[i] = weights[i]

    print('Class weights: {}'.format(class_weights))

    # do this check again vecause we need y_train to be 1-D for class weights
    if config["output_size"] > 1:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(config["train_dir"] +
                                                    model_name,
                                                    save_best_only=True)
    callbacks = [checkpoint]

    if config["tune"]:
        callbacks.append(
            MetricsReporterCallback(custom_validation_data=(x_val, y_val)))

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

    print(model.summary())

    print('Train...')
    model.fit(x_train,
              y_train,
              class_weight=class_weights,
              batch_size=config["batch_size"],
              epochs=config["max_epochs"],
              callbacks=callbacks,
              verbose=2,
              validation_data=(x_val, y_val))

    # #####
    # Evaluation time
    #
    _results = evaluate(model, test_data=(x_dev, y_dev), output_dict=True)

    logs = {
        # "accuracy" :_results["accuracy"],
        # "label0_precision" :_results["0"]["precision"],
        # "label0_recall" :_results["0"]["recall"],
        # "label0_f1_score" :_results["0"]["f1-score"],
        "label0_support" :_results["0"]["support"],
        "precision" :_results["1"]["precision"],
        "recall" :_results["1"]["recall"],
        # "label1_f1_score" :_results["1"]["f1-score"],
        "f1_score" :_results["1"]["f1-score"],
        "label1_support" :_results["1"]["support"],
        "macro_precision" :_results["macro avg"]["precision"],
        "macro_recall" :_results["macro avg"]["recall"],
        "macro_f1_score" :_results["macro avg"]["f1-score"],
        # "macro_support" :_results["macro avg"]["support"],
        "weighted_precision" :_results["weighted avg"]["precision"],
        "weighted_recall" :_results["weighted avg"]["recall"],
        "weighted_f1_score" :_results["weighted avg"]["f1-score"]}

    tune.track.log(keras_info=logs)

search_space = {
    "dropout":
        hp.uniform("dropout", 0.0, 0.9),
    "max_epochs":
        hp.choice("max_epochs", [10, 20, 30, 50]),
    "early_stop_delta":
        hp.choice("early_stop_delta", [0.001, 0.0001]),
    "early_stop_patience":
        hp.choice("early_stop_patience", [10, 20]),
    "hidden_activation":
        hp.choice("hidden_activation", ['tanh', 'relu', 'elu', 'selu']),
    "output_activation":
        hp.choice("output_activation", ['sigmoid', 'softmax']),
    "clipnorm":
        hp.choice("clipnorm", [0.5, 1.0, 2.5, 5.0, 10.0]),
    "learning_rate":
        hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-0)),
    "batch_size":
        hp.choice("batch_size", [20, 24, 32, 64, 128]),
    "nlayers":
        hp.randint('nlayers', 1, 5) * 1,
    "layer_size":
        hp.randint('layer_size', 1, 101) * 10,
}

_config.update({
    "hidden_activation": 'relu',
    "optimizer": 'adam',
    "threads": 4,
    "output_size": 2,
    "num_samples": 500
})

if not _config["tune"]:

    train_model(_config)

else:

    reporter = tune.CLIReporter()
    reporter.add_metric_column('keras_info/label1_f1_score', 'f1-score')

    ray.shutdown(
    )     # Restart Ray defensively in case the ray connection is lost.
    ray.init(num_cpus=6)
    results = tune.run(train_model,
                       name="tune-nn-bert-classifier",
                       config=_config,
                       stop={
                           "keras_info/f1_score": 0.99,
                           "training_iteration": 10**8
                       },
                       resources_per_trial={
                           "cpu": 1,
                           "gpu": 0
                       },
                       num_samples=_config["num_samples"],
                       checkpoint_freq=0,
                       checkpoint_at_end=False,
                       scheduler=AsyncHyperBandScheduler(time_attr='epoch',
                                                         metric='f1_score',
                                                         mode='max',
                                                         max_t=400,
                                                         grace_period=20),
                       search_alg=HyperOptSearch(search_space,
                                                 metric="keras_info/f1_score",
                                                 mode="max",
                                                 random_state_seed=SEED,
                                                 points_to_evaluate=[{
                                                     "dropout": 0.2,
                                                     "max_epochs": 2,
                                                     "early_stop_delta": 0,
                                                     "early_stop_patience": 0,
                                                     "hidden_activation": 1,
                                                     "output_activation": 0,
                                                     "clipnorm": 3,
                                                     "learning_rate": 0.0001,
                                                     "batch_size": 3,
                                                     "nlayers": 2,
                                                     "layer_size": 100
                                                 }]),
                       progress_reporter=reporter,
                       verbose=1)
    results.dataframe().to_csv('{0}/nn_results{1}layers.csv'.format(
        _config["train_dir"], _config['bert_tipe']))
