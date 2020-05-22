from sklearn.metrics import confusion_matrix, classification_report


def evaluate(model, test_data, perword=False, boosting=False, seq_lens=None):
    x_dev, y_dev = test_data

    if perword:
        return _evaluate_perword(model, x_dev, y_dev, seq_lens)

    return _evaluate_sentlevel(model, x_dev, y_dev, boosting)


def _evaluate_sentlevel(model, x_dev, y_dev, boosting):

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

    _results = classification_report(y_dev, y_pred, output_dict=True)

    return _results


def _evaluate_perword(model, x_dev, y_dev, seq_lens):

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

    _results = classification_report(y_true, y_pred, output_dict=True)

    return _results