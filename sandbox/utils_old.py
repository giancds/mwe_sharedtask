import os

import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight


def convert_flags_to_dict(flags):
    flags_as_dict = {}
    for key in flags:
        flags_as_dict[key] = flags[key].value
    return flags_as_dict


def build_model_name(name, FLAGS):
    _config = FLAGS
    if not isinstance(FLAGS, dict):
        _config = convert_flags_to_dict(FLAGS)
    name = (
        '{21}_{22}_{0}epochs.{1}-{2}eStop.{3}embDim.{4}-{5}dropout.{6}-{7}-{8}lstm.'
        '{9}lstmDrop.{10}lstmRecDrop.{11}-{12}.'
        '{14}Loss.{15}batch.{16}.{17}lr.{18}-{19}decay.{20}norm.'
        '{21}initScale.ckpt').format(
            _config["max_epochs"],
            _config["early_stop_patience"],
            _config["early_stop_delta"],
            _config["embed_dim"],
            _config["dropout"],
            'spatial-' if _config["spatial_dropout"] else '',
            _config["n_layers"],
            _config["lstm_size"],
            'bi-' if _config["bilstm"] else '',
            _config["lstm_dropout"],
            _config["lstm_recurrent_dropout"],
            _config["output_size"],
            _config["output_activation"],
            (str(_config["output_threshold"]) +
             'outThresh.') if _config["output_size"] == 1 and
            _config["output_activation"] == 'sigmoid' else '',
            _config["loss_function"],
            _config["batch_size"],
            _config["optimizer"],
            _config["learning_rate"],
            _config["lr_decay"],
            _config["start_decay"],
            _config["clipnorm"],
            _config["init_scale"],
            name,
            _config["feature"])
    print('\nModel name {}\n'.format(name))
    return name


def build_cnn_name(name, FLAGS):
    _config = FLAGS
    if not isinstance(FLAGS, dict):
        _config = convert_flags_to_dict(FLAGS)
    name = (
        '{20}_{21}_{0}epochs.{1}-{2}eStop.{3}embDim.{4}-{5}embDropout.'
        '{22}filters-{23}ngram.{24}pooling.{6}-{7}-{8}dense.'
        '{9}dropout.{10}-{11}.{12}Loss.{13}batch.{14}.{15}lr.{16}-{17}decay.'
        '{18}norm.{19}initScale.ckpt').format(
            _config["max_epochs"],
            _config["early_stop_patience"],
            '{:.2f}'.format(_config["early_stop_delta"]),
            _config["embed_dim"],
            _config["emb_dropout"],
            'spatial-' if _config["spatial_dropout"] else '',
            _config["n_layers"],
            _config["dense_size"],
            '{:.2f}'.format(_config["dropout"]),
            _config["output_size"],
            _config["output_activation"],
            (str(_config["output_threshold"]) +
                'outThresh.') if _config["output_size"] == 1 and
                _config["output_activation"] == 'sigmoid' else '',
            _config["loss_function"],
            _config["batch_size"],
            _config["optimizer"],
            '{:.5f}'.format(_config["learning_rate"]),
            '{:.2f}'.format(_config["lr_decay"]),
            _config["start_decay"],
            '{:.2f}'.format(_config["clipnorm"]),
            '{:.4f}'.format(_config["init_scale"]),
            '',
            _config["feature"],
            _config["filters"],
            _config["ngram"],
            _config["global_pooling"])
    print('\nModel name {}\n'.format(name))
    return name


def get_callbacks(FLAGS, model_name):

    checkpoint = tf.keras.callbacks.ModelCheckpoint(FLAGS.train_dir +
                                                    model_name,
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

    def lr_scheduler(epoch, lr): # pylint: disable=C0103
        lr_decay = FLAGS.lr_decay**max(epoch - FLAGS.start_decay, 0.0)
        return lr * lr_decay

    if FLAGS.start_decay > 0:
        lrate = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        callbacks.append(lrate)

    return callbacks


def get_optimizer(FLAGS):
    #
    # optimizers
    if FLAGS.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                             clipnorm=FLAGS.clipnorm)

    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=FLAGS.learning_rate, clipnorm=FLAGS.clipnorm)

    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate,
                                            clipnorm=FLAGS.clipnorm)

    return optimizer


def get_class_weights(weighted_loss, train_labels):
    class_weights = None

    if weighted_loss:
        weights = class_weight.compute_class_weight(
            'balanced', np.array([0, 1]), np.array([i for i in train_labels]))
        class_weights = {}

        for i in range(weights.shape[0]):
            class_weights[i] = weights[i]

    print('Class weights: {}'.format(class_weights))

    return class_weights


def define_rnn_flags(flags, base_dir, train_dir):
    #
    flags.DEFINE_integer("max_epochs", 100,
                         "Max number of epochs to train the models")

    flags.DEFINE_integer("n_estimators", 5,
                         "Number of models to train for the ensemble.")

    flags.DEFINE_float(
        "boost_lr", 1.0,
        "Learning rate for the contributions of each classifier.")

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

    flags.DEFINE_integer("embed_dim", 100, "Dimension of embbeddings.")

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

    flags.DEFINE_string(
        "optimizer", 'sgd',
        "Which optimizer to use. One of adam, sgd and rmsprop.")

    flags.DEFINE_float("learning_rate", 1.0, "Learning rate for the optimizer.")

    flags.DEFINE_float(
        "lr_decay", (1.0 / 1.15),
        "Rate to which we deca they learning rate during training.")

    flags.DEFINE_integer(
        "start_decay", 6,
        "Epoch to start the learning rate decay. To disable, set it to either 0 or to max_epochs"
    )

    flags.DEFINE_float("clipnorm", 5.0, "Max norm size to clipt the gradients.")

    flags.DEFINE_float("init_scale", 0.05,
                       "Range to initialize the weights of the model.")

    flags.DEFINE_integer("verbose", 2, "Verbosity of training")

    flags.DEFINE_string("feature", 'deprel',
                        "Which feature to use when training de model.")

    return flags.FLAGS


def define_cnn_flags(flags, base_dir, train_dir):
    #
    flags.DEFINE_integer("max_epochs", 100,
                         "Max number of epochs to train the models")

    flags.DEFINE_integer("n_estimators", 5,
                         "Number of models to train for the ensemble.")

    flags.DEFINE_float(
        "boost_lr", 1.0,
        "Learning rate for the contributions of each classifier.")


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

    flags.DEFINE_integer("embed_dim", 20, "Dimension of embbeddings.")

    flags.DEFINE_list("filters", [128], "Convolution filters.")

    flags.DEFINE_integer("ngram", 3, "N-gram size for the convolution.")

    flags.DEFINE_string(
        "global_pooling", 'max',
        "Type of global pooling applied at the end of convolutions.")

    flags.DEFINE_boolean("spatial_dropout", True,
                         "Whether or  to use spatial dropout for Embbeddings.")

    flags.DEFINE_float("emb_dropout", 0.1, "Embbeddings dropout.")

    flags.DEFINE_integer("dense_size", 50, "Dimension of LSTM layers.")

    flags.DEFINE_float("dropout", 0.1, "Dense regular dropout.")

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

    flags.DEFINE_float("init_scale", 0.05,
                       "Range to initialize the weights of the model.")

    flags.DEFINE_integer("verbose", 1, "Verbosity of training")

    flags.DEFINE_string("feature", 'upos+deprel',
                        "Which feature to use when training de model.")

    return flags.FLAGS