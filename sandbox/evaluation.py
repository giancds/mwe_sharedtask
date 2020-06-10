import numpy as np
import tensorflow as tf
from ray import tune
from sklearn.metrics import confusion_matrix, classification_report



class MetricsReporterCallback(tf.keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="epoch", logs=None, custom_validation_data=None):
        """Initializer.

        Args:
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        assert custom_validation_data, "validation_data should not be None"
        self.custom_validation_data = custom_validation_data
        self.iteration = 0
        logs = logs or {}
        # if freq not in ["batch", "epoch"]:
        #     raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = "epoch"
        super(MetricsReporterCallback, self).__init__()
        self._results = None
        self._batch_count = 0

    def on_batch_end(self, batch, logs=None):
        # from ray import tune
        # logs = logs or {}

        logs = self._update_logs(logs, predict=self.iteration == 0)

        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.report(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"))


    def on_epoch_end(self, batch, logs=None):

        print('Updating metrics')
        val_predict = (np.asarray(
            self.model.predict(self.custom_validation_data[0]))).round()

        self._results = classification_report(
            self.custom_validation_data[1], val_predict, output_dict=True)

        logs = self._update_logs(logs or {})

        if not self.freq == "epoch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            tune.track.log(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            tune.track.log(keras_info=logs, mean_accuracy=logs.get("accuracy"))

    def _update_logs(self, logs, predict=True):

        if self._results is None:

            logs.update({
                # "accuracy": 0.0accuracy"],
                # "label0_precision": 0.00,
                # "label0_recall": 0.00,
                # "label0_f1_score": 0.00,
                "label0_support": 0.00,
                "precision": 0.00,
                "recall": 0.0,
                # "label1_f1_score": 0.0,
                "f1_score": 0.0,
                "label1_support": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1_score": 0.0,
                # "macro_support": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1_score": 0.0})
        else:

            logs.update({
                # "accuracy" :self._results["accuracy"],
                # "label0_precision" :self._results["0"]["precision"],
                # "label0_recall" :self._results["0"]["recall"],
                # "label0_f1_score" :self._results["0"]["f1-score"],
                "label0_support" :self._results["0"]["support"],
                "precision" :self._results["1"]["precision"],
                "recall" :self._results["1"]["recall"],
                # "label1_f1_score" :self._results["1"]["f1-score"],
                "f1_score" :self._results["1"]["f1-score"],
                "label1_support" :self._results["1"]["support"],
                "macro_precision" :self._results["macro avg"]["precision"],
                "macro_recall" :self._results["macro avg"]["recall"],
                "macro_f1_score" :self._results["macro avg"]["f1-score"],
                # "macro_support" :self._results["macro avg"]["support"],
                "weighted_precision" :self._results["weighted avg"]["precision"],
                "weighted_recall" :self._results["weighted avg"]["recall"],
                "weighted_f1_score" :self._results["weighted avg"]["f1-score"]})

        return logs


def evaluate(model,
             test_data,
             perword=False,
             boosting=False,
             seq_lens=None,
             output_dict=False):
    x_dev, y_dev = test_data

    if perword:
        return _evaluate_perword(
            model, x_dev, y_dev, seq_lens, output_dict=output_dict)

    return _evaluate_sentlevel(
        model, x_dev, y_dev, boosting, output_dict=output_dict)


def _evaluate_sentlevel(model, x_dev, y_dev, boosting, output_dict=False):

    if boosting:
        y_pred = model.predict(x_dev)
    else:
        if model.layers[-1].output_shape[1] == 1:
            y_pred = model.predict(x_dev).astype('int')

        else:
            y_pred = model.predict(x_dev).argmax(axis=1)

    print('Confusion matrix:')
    print(confusion_matrix(y_dev, y_pred))

    print('\n\nReport')
    print(classification_report(y_dev, y_pred))

    _results = classification_report(y_dev, y_pred, output_dict=output_dict)

    return _results


def _evaluate_perword(model, x_dev, y_dev, seq_lens, output_dict=False):

    if model.layers[-1].output_shape[1] == 1:
        _y_pred = model.predict(x_dev).astype('int')
    else:
        _y_pred = model.predict(x_dev).argmax(axis=2)

    y_pred = []
    y_true = []
    for i, l in enumerate(seq_lens):
        y_pred += _y_pred[i, 0:l].tolist()
        y_true += y_dev[i, 0:l].tolist()

    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    print('\n\nReport')
    print(classification_report(y_true, y_pred))

    _results = classification_report(y_true, y_pred, output_dict=output_dict)

    return _results, y_pred