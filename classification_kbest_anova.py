# -*- coding: utf-8 -*-
"""
  Script to run feature selection and grid search for the best classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from classification_utils import get_training_test_set, report
from classification_utils import select_k_best_features

warnings.filterwarnings("ignore")

# pylint: disable=C0103

# reading the file from already merged csv
train_features, train_targets, test_features, test_targets, columns = get_training_test_set(
  "dublinbikes_data.csv", "dublinbikes_test.csv", numeric_only=True,
  remove_related=True, proportion=1)

# logistic regression
log = LogisticRegression()
svm = LinearSVC()  # LinearSVM
svc = SVC()  # RBF-SVM
knn = KNeighborsClassifier(metric="minkowski")  # K-NearestNeighbour
sgd = SGDClassifier(loss="hinge")  # SVm trained wth gradient descent
rdf = RandomForestClassifier()  # random forest
gnb = GaussianNB() # NaiveBayes

log_param_grid = {
  "penalty": ["l1", "l2"],
  "C": [1, 10, 100]
}

svm_param_grid = {
  "C": [1, 10, 100, 1000],
}

svc_param_grid = {
  "kernel": ["rbf"],
  "C": [1, 10, 100, 1000],
  "gamma": ["auto", 1e-2, 1e-3, 1e-4, 1e-5],
}

knn_param_grid = {
  "n_neighbors": [2, 3, 5, 10],
  "p": [1, 2],
  "weights": ["uniform", "distance"],
}

sgd_param_grid = {
  "loss": ["hinge"],
  "penalty": ["none", "l1", "l2", "elasticnet"],
  "alpha": [1e-2, 1e-3, 1e-4],
  "n_iter": [3, 5, 7, 10]
}

rdf_param_grid = {
  "n_estimators": [10, 25, 50, 75, 100],
  "criterion": ["gini", "entropy"],
  "max_features": ["sqrt", "log2"]
}

gnb_param_grid = {}

classifiers = [log, svm, svc, knn, sgd, rdf, gnb]
grid_params = [log_param_grid, svm_param_grid, svc_param_grid, knn_param_grid,
               sgd_param_grid, rdf_param_grid, gnb_param_grid]


results = OrderedDict()
results["classifier"] = []
results["n_features"] = []
results["method"] = []
results["mean"] = []
results["std_dev"] = []
results["matrix"] = []
results["accuracy"] = []
results["precision"] = []
results["recall"] = []
results["f1"] = []
results["parameters"] = []
results["features"] = []

metric = "f1"

for n in range(1, 101):

  print("Reducing to: {0} features\n".format(n))

  selected_features, selected_test_features, best = select_k_best_features(
    f_classif, train_features, test_features, train_targets, k_best=n)
  best_features_names = columns[best]

  for _, (classifier, params) in enumerate(zip(classifiers, grid_params)):

    print("Searching for: {0}\n".format(classifier.__class__.__name__))

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               cv=5,
                               n_jobs=-1,
                               scoring=metric)

    start = time.time()
    grid_search.fit(selected_features, train_targets)
    y_pred = grid_search.predict(selected_test_features)

    combinatons = len(grid_search.cv_results_["rank_test_score"])
    print("Search time: {:.2f} secs for {:d} candidates.".format(
          (time.time() - start), combinatons))
    mean, std, pars = report(grid_search.cv_results_)

    results["n_features"].append(n)
    results["method"].append("F-Anova")
    results["classifier"].append(classifier.__class__.__name__)
    results["parameters"].append(str(pars))
    results["mean"].append(mean)
    results["std_dev"].append(std)
    results["features"].append(str(best_features_names))

    acc = accuracy_score(test_targets, y_pred)
    prec, rec, f1_, _ = precision_recall_fscore_support(
      test_targets, y_pred, average="binary")
    matrix = confusion_matrix(test_targets, y_pred)

    results["matrix"].append(matrix)
    results["accuracy"].append(acc)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["f1"].append(f1_)

idx = [i for i in range(1, len(results["n_features"]) + 1)]
results_df = pd.DataFrame(results, index=idx)

columns = ["classifier", "n_features", "mean", "matrix",
           "accuracy", "precision", "recall", "f1"]
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_columns", len(columns))
print(results_df.sort("f1", ascending=False)[columns])

results_df.to_csv("results_kbest_anova.csv", sep=",")
