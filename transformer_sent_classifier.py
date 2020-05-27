#pylint: disable=invalid-name

import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

tf.compat.v1.flags.DEFINE_string("model_type",
                                 'distilbert-base-multilingual-cased',
                                 "Model type to extract the embeddings")

tf.compat.v1.flags.DEFINE_string(
    "language_code", 'all',
    "Language code to use for fitting and evaluating the classifiers. "
    "One of ['DE', 'GA', 'HI', 'PT', 'ZH']. Set to 'all' to use all.")

FLAGS = tf.compat.v1.flags.FLAGS

codes = (['DE', 'GA', 'HI', 'PT', 'ZH']
         if FLAGS.language_code is 'all' else [FLAGS.language_code])

SEED = 42

model_type = 'distilbert-base-multilingual-cased'
with open('data/{}.embdata.pkl'.format(FLAGS.model_type), 'rb') as f:
    data = pickle.load(f)

x_train = np.concatenate([data[code]['x_train'] for code in codes], axis=0)
y_train = np.concatenate([data[code]['y_train'] for code in codes], axis=0)
print(x_train.shape, y_train.shape)

x_dev = np.concatenate([data[code]['x_train'] for code in codes], axis=0)
y_dev = np.concatenate([data[code]['y_train'] for code in codes], axis=0)
print(x_dev.shape, y_dev.shape)

del data


def eval(_y_true, _y_pred):
    print(confusion_matrix(_y_true, _y_pred))
    print(classification_report(_y_true, _y_pred))


print('KNeighborsClassifier')
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_dev)
eval(y_dev, y_pred)

print('LinearSVC')
linear_svc = LinearSVC(dual=False, class_weight='balanced')
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_dev)
eval(y_dev, y_pred)

print('SVC')
svc = SVC(class_weight='balanced')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_dev)
eval(y_dev, y_pred)

print('SGDClassifier')
sgd = SGDClassifier(class_weight='balanced')
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_dev)
eval(y_dev, y_pred)

print('AdaBoostClassifier')
ada = AdaBoostClassifier()
ada.fit(x_train, y_train)
y_pred = ada.predict(x_dev)
eval(y_dev, y_pred)