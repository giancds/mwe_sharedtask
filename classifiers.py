import copy
import numpy as np
import tensorflow as tf


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
        res.update({'build_fn': self.build_fn,
                    'callbacks': self.callbacks,
                    'verbose': self.verbose,
                    'validation_data': self.validation_data})
        return res


    def fit(self, x, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
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
            kwargs['sample_weight'] = sample_weight
        if self.output_size > 1:
            y = tf.keras.utils.to_categorical(y)
        return super(BoostedClassifier, self).fit(x, y, **kwargs)