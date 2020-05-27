import os

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from ray import tune
from sklearn.metrics import confusion_matrix, classification_report


def convert_flags_to_dict(flags):
    flags_as_dict = {}
    for key in flags:
        flags_as_dict[key] = flags[key].value
    return flags_as_dict


def build_model_name(FLAGS):
    _config = FLAGS
    if not isinstance(FLAGS, dict):
        _config = convert_flags_to_dict(FLAGS)
    _name = (
        'nn_classifier.{0}.{1}.{2}epochs.{3}-{4}eStop.{5}-{6}layers-{7}dropout.'
        '{8}-{9}.output.{10}.thresh.{11}{12}Loss.{13}batch.{14}.{15}lr.'
        '{16}-{17}decay.{18}norm.ckpt').format(
            _config['bert_type'], # 0
            _config['language_code'],  # 1
            _config['max_epochs'],  # 2
            _config['early_stop_patience'],  # 3
            _config['early_stop_delta'],  # 4
            str(_config['layers']),  # 5{14}Loss.{15}batch.
            _config['hidden_activation'],  # 6
            _config['dropout'],  # 7
            _config['output_size'],  # 8
            _config['output_activation'],  # 9
            _config['output_threshold'],  # 10
            'weighted.' if _config['weighted_loss'] else '', # 11
            _config['loss_function'],  # 12
            _config['batch_size'],  # 13
            _config['optimizer'],  # 14
            _config['learning_rate'],  # 15
            _config['lr_decay'],  # 16
            _config['start_decay'],  # 17
            _config['clipnorm'])   # 18
    print('\nModel name {}\n'.format(_name))
    return _name


def define_nn_flags(flags, base_dir, train_dir):

    flags.DEFINE_boolean("tune", True,
                         "Whether or not to tune hyperparameters.")

    flags.DEFINE_integer(
        "num_samples", 20,
        "How many training steps to monitor. Set to 0 to ignore.")

    flags.DEFINE_string(
        "bert_type",
        'distilbert-base-multilingual-cased',
        "Model type to extract the embeddings")

    flags.DEFINE_string(
        "language_code", 'all',
        "Language code to use for fitting and evaluating the classifiers. "
        "One of ['DE', 'GA', 'HI', 'PT', 'ZH']. Set to 'all' to use all.")
    #
    flags.DEFINE_integer("max_epochs", 100,
                         "Max number of epochs to train the models")

    flags.DEFINE_integer(
        "early_stop_patience", 10,
        "How many training steps to monitor. Set to 0 to ignore.")

    flags.DEFINE_float(
        "early_stop_delta", 0.001,
        "How many training steps to monitor. Set to 0 to ignore.")

    flags.DEFINE_boolean("log_tensorboard", False,
                         "Whether or not to log info using tensorboard")

    flags.DEFINE_string("train_dir",
                        os.path.join(base_dir, train_dir) + "/",
                        "Train directory")

    flags.DEFINE_list("layers", [100, 100], "Dense layers.")

    flags.DEFINE_string("hidden_activation", 'sigmoid',
                        "Activation for the output layer.")


    flags.DEFINE_float("dropout", 0.1, "Dense regular dropout.")


    flags.DEFINE_string("output_activation", 'sigmoid',
                        "Activation for the output layer.")

    flags.DEFINE_integer(
        "output_size", 2,
        "Size of the output layer. Only relevant when using sigmoid output.")

    flags.DEFINE_float(
        "output_threshold", 0.5,
        "Threshold to classify a sentence as idiomatic or not. "
        "Only relevant when using a single sigmoid output."
    )

    flags.DEFINE_string("loss_function", 'binary_crossentropy',
                        "Loss function to use during training.")

    flags.DEFINE_boolean("weighted_loss", True,
                         "Whether or to use weighted loss for learning.")

    flags.DEFINE_integer("batch_size", 32, "Size of batches.")

    flags.DEFINE_string(
        "optimizer", 'sgd',
        "Which optimizer to use. One of adam, sgd and rmsprop.")

    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for the optimizer.")

    flags.DEFINE_float(
        "lr_decay", (1.0 / 1.15),
        "Rate to which we deca they learning rate during training.")

    flags.DEFINE_integer(
        "start_decay", 0,
        "Epoch to start the learning rate decay. To disable, set it to either 0 or to max_epochs"
    )

    flags.DEFINE_float("clipnorm", 5.0, "Max norm size to clipt the gradients.")

    flags.DEFINE_integer("verbose", 1, "Verbosity of training")


    return flags.FLAGS