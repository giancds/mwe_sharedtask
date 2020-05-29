#pylint: disable=invalid-name
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp


SEED = 42
BASE_DIR = os.path.expanduser("~")     # this will point to the user's home
TRAIN_DIR = "ray_results"


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

# knn = KNeighborsClassifier(n_jobs=-1)
linear_svc = LinearSVC(dual=False, class_weight='balanced', random_state=SEED)
# svc = SVC(class_weight='balanced', random_state=SEED)
# sgd = SGDClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED)
# ada = AdaBoostClassifier(random_state=SEED)


def fit_and_score(clf, n_samples):

    scores = cross_val_score(
        clf, x_train, y_train, cv=n_samples, scoring='f1')

    score_log = {'f1-mean': scores.mean()}
    for i, score in enumerate(scores):
        score_log['f1-cv{}'.format(i+1)] = score

    tune.track.log(cv_score=score_log)


def search_knn(_config):

    clf = KNeighborsClassifier(
        n_jobs=-1,
        n_neighbors=_config['n_neighbors'],
        weights=_config['weights'],
        p=_config['p'])
    fit_and_score(clf, _config['n_samples'])


def search_svm(_config):

    clf = LinearSVC(
        penalty=_config['penalty'],
        loss=['loss'],
        C=_config['C'])
    fit_and_score(clf, _config['n_samples'])


def search_lreg(_config):

    clf = LogisticRegression(
        n_jobs=-1,
        penalty=_config['penalty'],
        C=_config['C'])
    fit_and_score(clf, _config['n_samples'])


config = {
    "n_samples": 3,
    "n_trials": 1
}

knn_space = {
    'n_neighbors': hp.randint('n_neighbors', 1, 10),
    'weights':  hp.choice('weights',['uniform', 'distance']),
    'p':  hp.choice('p', [1, 2])
}

svm_space = {
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'loss':  hp.choice('loss', ['hinge', 'squared_hinge']),
    'C':  hp.choice('C', [0.001, 0.1, 1.0, 10.0, 100.0])
}

lreg_space = {
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'C':  hp.choice('C', [0.001, 0.1, 1.0, 10.0, 100.0])
}

results = tune.run_experiments(
        tune.Experiment(
            run=search_lreg,
            name="tune-nn-bert-classifier-lr",
            config=config,
            stop={"cv_score/f1-mean": 0.99,},
            # resources_per_trial={"cpu": 4},
            num_samples=config["n_trials"]),
        scheduler=AsyncHyperBandScheduler(
            time_attr='training_iteration',
            metric="cv_score/f1-mean",
            mode='max'),
        search_alg=HyperOptSearch(
            lreg_space,
            metric="cv_score/f1-mean",
            mode="max",
            random_state_seed=SEED),
        verbose=1)
try:
    df = pd.read_csv(TRAIN_DIR + '/results.csv')

except FileNotFoundError as e:
    print(e)

results = tune.run_experiments(
        tune.Experiment(
            run=search_lreg,
            name="tune-nn-bert-classifier-lr",
            config=config,
            stop={"cv_score/f1-mean": 0.99,},
            # resources_per_trial={"cpu": 4},
            num_samples=config["n_trials"]),
        scheduler=AsyncHyperBandScheduler(
            time_attr='training_iteration',
            metric="cv_score/f1-mean",
            mode='max'),
        search_alg=HyperOptSearch(
            lreg_space,
            metric="cv_score/f1-mean",
            mode="max",
            random_state_seed=SEED),
        verbose=1)
try:
    df = pd.read_csv(TRAIN_DIR + '/results.csv')

except FileNotFoundError as e:
    print(e)
    results.dataframe().to_csv(TRAIN_DIR + '/results.csv')










