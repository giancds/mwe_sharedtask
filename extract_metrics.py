import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

home = "/Users/gian/Google Drive/mwe_sharedtask/data"
data_file = "distilbert-base-multilingual-cased.multilabel.tokenized__.pkl"

language_codes = ['DE', 'GA', 'HI', 'PT', 'ZH']

y_true, y_pred = {}, {}

with open('{}/{}'.format(home, data_file), 'rb') as f:
    data = pickle.load(f)

for code in language_codes:
    y_true[code] = [i for j in data[code]['y_test'] for i in j]
    y_pred[code] = [i for j in data[code]['y_sys'] for i in j]
    diff = [0] * (len(y_true[code]) - len(y_pred[code]))
    y_pred[code] += diff
    print(confusion_matrix(y_true[code], y_pred[code]))
    print(classification_report(y_true[code], y_pred[code]))