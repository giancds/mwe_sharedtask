import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = BASE_DIR +  "/ray_results"


model_type = 'distilbert-base-multilingual-cased'
# model_type = 'bert-base-multilingual-cased'
with open('data/{}.embdata.pkl'.format(model_type), 'rb') as f:
    data = pickle.load(f)

codes = ['DE', 'GA', 'HI', 'PT', 'ZH']


x_train = np.concatenate([data[code]['x_train'] for code in codes], axis=0)
y_train = np.concatenate([data[code]['y_train'] for code in codes], axis=0)
print(x_train.shape, y_train.shape)

x_dev = np.concatenate([data[code]['x_dev'] for code in codes], axis=0)
y_dev = np.concatenate([data[code]['y_dev'] for code in codes], axis=0)
print(x_dev.shape, y_dev.shape)

del data


knn_space = {
    'model': [KNeighborsClassifier(n_jobs=-1)],
    'model__n_neighbors': Integer(1, 10),
    'model__weights':  Categorical(['uniform', 'distance']),
    'model__p':  Integer(1, 2)
}

lsvm_space = {
    'model': [LinearSVC(dual=False, class_weight='balanced', random_state=SEED)],
    'model__penalty': Categorical(['l1', 'l2']),
    'model__loss':  Categorical(['hinge', 'squared_hinge']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
}


svm_space = {
    'model': [SVC(class_weight='balanced', random_state=SEED)],
    'model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'model__degree': Integer(1, 3),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
}

lreg_space = {
    'model': [LogisticRegression(solver='liblinear')],
    'model__penalty': Categorical(['l1', 'l2']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
}


pipe = Pipeline([
    ('model', LogisticRegression(solver='liblinear'))
])

opt = BayesSearchCV(
   pipe,
    [(knn_space, 20), (lsvm_space, 20), (lreg_space, 20)], # (parameter space, # of evaluations)
    n_iter=32,
    cv=3,
    n_jobs=-1,
    random_state=SEED,
    verbose=2
)

opt.fit(x_train, y_train)