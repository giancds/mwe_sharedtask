import pickle

import numpy as np
from sklearn.model_selection import train_test_split

features = {
    '*': 0,
    'IAV': 1,
    'IRV': 2,
    'LVC.cause': 3,
    'LVC.full': 4,
    'MVC': 5,
    'VID': 6,
    'VPC.full': 7,
    'VPC.semi': 8,
}


def load_tokenized_data(datafile,
                        language_codes,
                        percent=1.0,
                        seed=42,
                        binary=False,
                        split=True):

    with open(datafile, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_dev, y_dev = {}, {}
    for code in language_codes:

        true_x, true_y = [], []
        false_x, false_y = [], []
        for i, (xsample, ysample) in enumerate(
                zip(data[code]['x_train'], data[code]['y_train'])):

            if sum(ysample) > 0:
                true_x.append(xsample)
                if binary:
                    ysample = [0 if y == 0 else 1 for y in ysample]
                true_y.append(ysample)

        max_len = max([len(y) for y in true_y])
        for xsample, ysample in zip(data[code]['x_train'],
                                    data[code]['y_train']):
            if sum(ysample) == 0 and len(ysample) < max_len:
                false_x.append(xsample)
                false_y.append(ysample)

        false_x = np.array(false_x)
        false_y = np.array(false_y)

        np.random.seed(seed)
        idx = np.random.randint(len(false_y), size=int(percent * len(true_y)))
        false_x = false_x[idx].tolist()
        false_y = false_y[idx].tolist()

        x_train += true_x + false_x
        y_train += true_y + false_y


    if split:
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          random_state=seed,
                                                          test_size=0.15,
                                                          shuffle=True)
    else:
        x_val, y_val = [], []
        for code in language_codes:
            x_val += data[code]["x_dev"]
            y_val += data[code]["y_dev"]

        x_dev[code] = data[code]["x_dev"]
        y_dev[code] = data[code]["y_dev"]

    del data

    return (x_train, y_train), (x_val, y_val), (x_dev, y_dev)
