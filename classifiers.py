import copy
import numpy as np
import tensorflow as tf

import types
from keras.utils.np_utils import to_categorical
from keras import losses


class BoostedClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):

    def __init__(self,
                 build_fn=None,
                 callbacks=None,
                 verbose=2,
                 validation_data=None,
                 output_size=2,
                 **sk_params):

        self.build_fn = build_fn
        self.callbacks = callbacks
        self.verbose = verbose
        self.validation_data = validation_data
        self.output_size = output_size

        sk_params['callbacks'] = callbacks
        sk_params['verbose'] = verbose
        sk_params['validation_data'] = validation_data

        super(BoostedClassifier, self).__init__(build_fn, **sk_params)

    def get_params(self, deep=True):
        res = copy.deepcopy(self.sk_params)
        res.update({
            'build_fn': self.build_fn,
            'callbacks': self.callbacks,
            'verbose': self.verbose,
            'validation_data': self.validation_data
        })
        return res

    def fit(self, x, y, sample_weight=None, **kwargs):
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        if sample_weight is not None:
            if len(self.validation_data[1].shape) == 3:
                sample_weight = sample_weight.flatten()
            kwargs['sample_weight'] = sample_weight
        if self.output_size > 1:
            y = tf.keras.utils.to_categorical(y)
        return super(BoostedClassifier, self).fit(x, y, **kwargs)


class BoostedTemporalClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):

    def __init__(self,
                 build_fn=None,
                 callbacks=None,
                 verbose=2,
                 validation_data=None,
                 output_size=2,
                 **sk_params):

        self.build_fn = build_fn
        self.callbacks = callbacks
        self.verbose = verbose
        self.validation_data = validation_data
        self.output_size = output_size

        sk_params['callbacks'] = callbacks
        sk_params['verbose'] = verbose
        sk_params['validation_data'] = validation_data

        self.classes_ = np.unique(
            [i for i in validation_data[1].flatten() if i >= 0])

        print('\n\nClasses {}\n\n'.format(self.classes_))


        self.n_classes_ = len(self.classes_)

        print('\n\nN Classes {}\n\n'.format(self.n_classes_))

        super(BoostedTemporalClassifier, self).__init__(build_fn, **sk_params)

    def get_params(self, deep=True):
        res = copy.deepcopy(self.sk_params)
        res.update({
            'build_fn': self.build_fn,
            'callbacks': self.callbacks,
            'verbose': self.verbose,
            'validation_data': self.validation_data
        })
        return res

    def fit(self, x, y, sample_weight=None, **kwargs):

        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        y = np.array(y)
        x = x.reshape(-1, self.validation_data[1].shape[1])
        y = y.reshape(-1, self.validation_data[1].shape[1])
        y = np.expand_dims(y, axis=2)
        self.n_classes_ = 2
        if sample_weight is not None:
            sample_weight = sample_weight.reshape(
                -1, self.validation_data[1].shape[1])
            # kwargs['sample_weight'] = np.expand_dims(sample_weight, axis=2)
            kwargs['sample_weight'] = sample_weight

        if (losses.is_categorical_crossentropy(self.model.loss) and
                len(y.shape) != 2):
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(tf.keras.Sequential.fit))
        fit_args.update(kwargs)

        print('\n\nx shape {0}\n\n'.format(x.shape))
        history = self.model.fit(x, y, **fit_args)

        return history

    def predict_proba(self, x, **kwargs):

        kwargs = self.filter_sk_params(tf.keras.Sequential.predict_proba,
                                       kwargs)
        probs = self.model.predict_proba(
            x.reshape(-1, self.validation_data[1].shape[1]), **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs
