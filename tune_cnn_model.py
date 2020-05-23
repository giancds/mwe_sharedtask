import os

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from preprocess import Features, load_dataset, pre_process_data
from evaluation import evaluate
from utils import get_callbacks, get_optimizer
from utils import define_cnn_flags, build_cnn_name, convert_flags_to_dict

import ray
from ray import tune
from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import Repeater
from hyperopt import hp

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# pylint: disable=W0613,C0103,C0112

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

#
SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "train_mwe_classifier"

# #####
# Some hyperparameter definitions
#
FLAGS = define_cnn_flags(tf.compat.v1.flags, BASE_DIR, TRAIN_DIR)

# define which feature we can use to train de model

model_name = build_cnn_name('sentlevel_cnn', FLAGS)

FLAGS.filters = [int(i) for i in FLAGS.filters]
# #####
# Loading data
#

_config = convert_flags_to_dict(FLAGS)
_config["is_dev"] = True

cwd = os.getcwd()


print('Pre-processing data...')
tmp = _config["feature"].split('+')
features = []
for f in tmp:
    if f == 'upos':
        features.append(Features.upos)

    elif f == 'deprel':
        features.append(Features.deprel)

train_files = [cwd + '/data/GA/train.cupt'] if _config["is_dev"] else []
train_sents, train_labels = load_dataset(
    train_files, features, cnn=True)

# validation/dev dataset
dev_files = [cwd + '/data/GA/dev.cupt'] if _config["is_dev"] else []
dev_sents, dev_labels = load_dataset(
    dev_files, features, cnn=True, train=False)

train_data, dev_data, (max_len, n_tokens) = pre_process_data(
    (train_sents, train_labels), (dev_sents, dev_labels),
    seed=SEED, cnn=True)

_x_train, _x_val, _y_train, _y_val = train_data
_x_dev, _y_dev = dev_data

_config["x_train"] = _x_train
_config["x_val"] = _x_val
_config["y_train"] = _y_train
_config["y_val"] = _y_val

_config["x_dev"] = _x_dev
_config["y_dev"] = _y_dev

_config["n_tokens"] = n_tokens
_config["max_len"] = _y_dev



def train_model(config):


    print("Building model...")

    model = tf.keras.Sequential()
    # embedding
    model.add(
        tf.keras.layers.Embedding(
            config["n_tokens"] + 1,
            config["embed_dim"],
            input_shape=(config["x_train"].shape[1], config["x_train"].shape[2]),
            input_length=config["max_len"],
            mask_zero=True,
            embeddings_initializer=tf.random_uniform_initializer(
                minval=-config["init_scale"], maxval=config["init_scale"],
                seed=SEED)))

    shape = model.layers[0].output_shape

    model.add(tf.keras.layers.Reshape((shape[1], shape[3], shape[2])))

    if config["spatial_dropout"]:
        model.add(tf.keras.layers.SpatialDropout2D(config["emb_dropout"]))
    else:
        model.add(tf.keras.layers.Dropout(config["emb_dropout"]))

    for filters in config["filters"]:
        model.add(
            tf.keras.layers.Conv2D(
                filters,
                config["ngram"],
                padding='valid',
                activation='relu',
            #    strides=1,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-config["init_scale"], maxval=config["init_scale"],
                    seed=SEED)))
        model.add(tf.keras.layers.MaxPooling2D())

    if config["global_pooling"] == 'average':
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif config["global_pooling"] == 'max':
        model.add(tf.keras.layers.GlobalMaxPool2D())

    for _ in range(config["n_layers"]):
        model.add(
            tf.keras.layers.Dense(
                config["dense_size"],
                activation='relu',
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-config["init_scale"],
                    maxval=config["init_scale"],
                    seed=SEED)))
        model.add(tf.keras.layers.Dropout(config["dropout"]))

    model.add(tf.keras.layers.Flatten())

    if config["output_size"] == 1:
        model.add(
            tf.keras.layers.Dense(1,
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
                2, activation=config["output_activation"],
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
    callbacks = [checkpoint]

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

    def lr_scheduler(epoch, lr): # pylint: disable=C0103
        lr_decay = config["lr_decay"]**max(epoch - config["start_decay"], 0.0)
        return lr * lr_decay

    if config["start_decay"] > 0:
        lrate = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        callbacks.append(lrate)


    print('Train...')
    model.fit(
        config["x_train"],
        y_train,
        class_weight=class_weights,
        batch_size=config["batch_size"],
        epochs=config["max_epochs"],
        callbacks=callbacks,
        verbose=config["verbose"],
        validation_data=(config["x_val"], y_val))

    # #####
    # Evaluation time
    #
    _results = evaluate(model, test_data=(config["x_dev"], config["y_dev"]))

    # tune.track.log(
    #     accuracy=_results["accuracy"],
    #     label0_precision=_results["0"]["precision"],
    #     label0_recall=_results["0"]["recall"],
    #     label0_f1_score=_results["0"]["f1-score"],
    #     label0_support=_results["0"]["support"],
    #     label1_precision=_results["1"]["precision"],
    #     label1_recall=_results["1"]["recall"],
    #     label1_f1_score=_results["1"]["f1-score"],
    #     label1_support=_results["1"]["support"],
    #     macro_precision=_results["macro avg"]["precision"],
    #     macro_recall=_results["macro avg"]["recall"],
    #     macro_f1_score=_results["macro avg"]["f1-score"],
    #     macro_support=_results["macro avg"]["support"],
    #     weighted_precision=_results["weighted avg"]["precision"],
    #     weighted_recall=_results["weighted avg"]["recall"],
    #     weighted_f1_score=_results["weighted avg"]["f1-score"],
    #     weighted_support=_results["weighted avg"]["support"])

    return _results


search_space = {
    "embed_dim": hp.choice("embed_dim", [10, 20, 30, 50, 75, 100]),
    "emb_dropout": hp.uniform("emb_dropout", 0.0, 0.9),
    "dropout": hp.uniform("dropout", 0.0, 0.9),
    "spatial_dropout": hp.choice("spatial_dropout", [True, False]),
    "init_scale": hp.loguniform("init_scale", np.log(1e-2), np.log(1e-1)),
    "n_layers": hp.choice("n_layers", [1, 2, 3, 4, 5]),
    "dense_size": hp.choice("dense_size", [10, 20, 30, 50, 75, 100]),
    "max_epochs": hp.choice("max_epochs", [10, 20, 30, 50]),
    "early_stop_delta": hp.choice("early_stop_delta", [0.001, 0.0001]),
    "early_stop_patience": hp.choice("early_stop_patience", [10, 20]),
    "optimizer": hp.choice("optimizer", ['sgd', 'adam', 'rmsprop']),
    "output_activation": hp.choice("output_activation", ['sigmoid', 'softmax']),
    "feature":  hp.choice("feature", ["upos+deprel", "upos", "deprel"]),
    "filters": hp.choice("filters", [[128], [128, 64, 32], [64, 32], [128, 32]]),
    "ngram": hp.choice("ngram", [1, 2, 3, 4, 5]),
    "global_pooling": hp.choice("global_pooling", ['', 'max', 'average']),
    "clipnorm": hp.choice("clipnorm", [0.5, 1.0, 2.5, 5.0, 10.0]),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-0)),
    "optimizer": hp.choice("optimizer", ['sgd', 'adam', 'rmsprop']),
    "batch_size": hp.choice("batch_size", [20, 24, 32, 64, 128]),
}


_config.update({
    "threads": 2,
    "output_size": 2
})

results = tune.run_experiments(
    tune.Experiment(
        run=train_model,
        name="tune-cnn",
        config=_config,
         stop={
            "keras_info/label1_f1_score": 0.99,
            "training_iteration": 10**8
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        num_samples=10,
        checkpoint_freq=0,
        checkpoint_at_end=False),
    scheduler=AsyncHyperBandScheduler(
        time_attr='epoch',
        metric='keras_info/label1_f1_score',
        mode='max',
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
            "spatial_dropout": 0,
            "init_scale": 0.05,
            "n_layers": 0,
            "dense_size": 3,
            "max_epochs": 0,
            "early_stop_delta": 0,
            "early_stop_patience": 0,
            "optimizer": 1,
            "output_activation": 0,
            "feature":  0,
            "filters": 0,
            "ngram": 2,
            "global_pooling": 1,
            "clipnorm": 1,
            "learning_rate": 0.0001,
            "optimizer": 1,
            "batch_size": 2,
        }]))
results.dataframe().to_csv(_config["train_dir"] + '/results.csv')
