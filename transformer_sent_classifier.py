#pylint: disable=invalid-name
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

import ray
# from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

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


def run_tune(fit_function, space):

    results = ray.tune.run(
        fit_function,
        name="tune-nn-bert-classifier-lr",
        config={},
        stop={"cv_score/f1-mean": 0.90,},
        num_samples=10,
        scheduler=AsyncHyperBandScheduler(
            time_attr='training_iteration',
            metric="cv_score/f1-mean",
            mode='max'),
        search_alg=HyperOptSearch(
            space,
            metric="cv_score/f1-mean",
            mode="max",
            random_state_seed=SEED),
        verbose=1)
    df = results.dataframe()
    try:
        df0 = pd.read_csv('{}/results.csv'.format(TRAIN_DIR))
        df = pd.concat([df0, df])
    except FileNotFoundError as e:
        print(e)
    df.to_csv('{}/results.csv'.format(TRAIN_DIR))


def fit_and_score(clf):

    scores = cross_val_score(
        clf, x_train, y_train, cv=3, scoring='f1')

    score_log = {'f1-mean': scores.mean()}
    for i, score in enumerate(scores):
        score_log['f1-cv{}'.format(i+1)] = score

    ray.tune.track.log(cv_score=score_log)


def fit_knn(config):
    fit_and_score(KNeighborsClassifier(**config))


def fit_linear_svm(config):
    fit_and_score(LinearSVC(**config))


def fit_svm(config):
    fit_and_score(SVC(**config))


def fit_lregression(config):
    fit_and_score(LogisticRegression(**config))


def fit_adaboost(config):
    fit_and_score(AdaBoostClassifier(**config))


def fit_histgrad(config):
    fit_and_score(HistGradientBoostingClassifier(**config))


def fit_rforest(config):
    fit_and_score(RandomForestClassifier(**config))


knn_space = {
    'n_neighbors': hp.randint('n_neighbors', 1, 10),
    'weights':  hp.choice('weights', ['uniform', 'distance']),
    'p':  hp.randint('p', 1, 2) * 1,
    'n_jobs': hp.choice('n_jobs', [-1])
}

lsvm_space = {
    'penalty':hp.choice('penalty', ['l1', 'l2']),
    'loss':  hp.choice('loss', ['hinge', 'squared_hinge']),
    'C': hp.loguniform('C', np.log(1e-6), np.log(1e+4)),
    'class_weight': hp.choice('class_weight', ['balanced']),
    'random_state': hp.choice('random_state', [SEED]),
    'dual': hp.choice('dual', [True, False])
}

svm_space = {
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': hp.randint('degree', 1, 3) * 1,
    'gamma': hp.choice('gamma', ['scale', 'auto']),
    'C': hp.loguniform('C', np.log(1e-6), np.log(1e+4)),
    'class_weight': hp.choice('class_weight', ['balanced']),
    'random_state': hp.choice('random_state', [SEED]),
}

lreg_space = {
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'C': hp.loguniform('C', np.log(1e-6), np.log(1e+4)),
    'solver': hp.choice('solver', ['liblinear']),
    'n_jobs': hp.choice('n_jobs', [-1])
}

adab_space = {
    'base_estimator': hp.choice('base_estimator', [DecisionTreeClassifier(max_depth=hp.randint('max_depth', 1, 10))]),
    'n_estimators': hp.randint('n_estimators', 1, 100) * 5,
    'learning_rate': hp.uniform('learning_rate', 0.1, 1.0),
    'random_state': hp.choice('random_state', [SEED]),
}

hist_space = {
    'learning_rate': hp.uniform('learning_rate', 0.1, 1.0),
    'max_leaf_nodes': hp.randint('max_leaf_nodes', 2, 100) * 1,
    'max_depth': hp.randint('max_depth', 2, 100) * 1,
    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 50) * 1,
    'l2_regularization': hp.uniform('l2_regularization', 0.0, 10.0),
    'max_bins': hp.randint('max_bins', 1, 60) * 5,
    'random_state': hp.choice('random_state', [SEED]),
}

rf_space = {
    'n_estimators': hp.randint('n_estimators', 1, 100) * 5,
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_leaf_nodes': hp.randint('max_leaf_nodes', 2, 100) * 1,
    'max_depth': hp.randint('max_depth', 2, 100) * 1,
    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 50) * 1,
    'min_samples_split': hp.randint('min_samples_split', 1, 10) * 1,
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    'random_state': hp.choice('random_state', [SEED]),
    'n_jobs': hp.choice('n_jobs', [-1])
}

ray.shutdown()
ray.init()


run_tune(fit_knn, knn_space)
run_tune(fit_linear_svm, lsvm_space)
run_tune(fit_svm, svm_space)
run_tune(fit_lregression, lreg_space)
run_tune(fit_adaboost, adab_space)
run_tune(fit_histgrad, hist_space)
run_tune(fit_rforest, rf_space)


ray.shutdown()
