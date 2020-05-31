# -*- coding: utf-8 -*-
"""
  Script to run a calibration of the classifiers to get probabilistic
    predictions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import pandas as pd


from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from classification_utils import get_training_test_set, get_remaining_data
from classification_utils import select_k_best_features

warnings.filterwarnings("ignore")

# pylint: disable=C0103

trainfile, testfile = "dublinbikes_data.csv", "dublinbikes_test.csv"

# reading the file from already merged csv
train_features, train_targets, test_features, test_targets, columns = get_training_test_set(
  trainfile, testfile, numeric_only=True,
  remove_related=True, proportion=1)

remaining, rem_features, _ = get_remaining_data(trainfile, testfile)

# building the classifier with the selected parameters
rdf = RandomForestClassifier(n_estimators=25,
                             max_features="sqrt",
                             criterion="entropy",
                             random_state=136)
n_features = 10

# obtaining the n_features best features
selector = SelectKBest(chi2, k=n_features).fit(train_features, train_targets)
selected_features = selector.transform(train_features)
selected_test_features = selector.transform(test_features)
selected_rem_features = selector.transform(rem_features)

# sanity check - fit the randomforest and get the test metrics
rdf.fit(selected_features, train_targets)
y_pred = rdf.predict(selected_test_features)
acc = accuracy_score(test_targets, y_pred)
prec, rec, f1_, _ = precision_recall_fscore_support(
  test_targets, y_pred, average="binary")
matrix = confusion_matrix(test_targets, y_pred)
print("Acc.: {0} \nPrec.: {1} \nRec.: {2} \nf1: {3} \nMatrix: {4}".format(
  acc, prec, rec, f1_, matrix
))

# now calibrating the classifier
calibrated_rdf = CalibratedClassifierCV(rdf, cv=5, method="sigmoid")
calibrated_rdf.fit(selected_features, train_targets)

prob_pos_clf = calibrated_rdf.predict_proba(selected_rem_features)
remaining["prob_no"] = prob_pos_clf[0:,0]
remaining["prob_yes"] = prob_pos_clf[0:,1]

columns = ["GEOGID", "ED_Ward", "prob_yes"]

pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_columns", len(columns))
pd.set_option("display.max_rows", 60)

print(remaining.sort("prob_yes", ascending=False)[columns])

most_likely = remaining.sort("prob_yes", ascending=False).head(30)
least_likely = remaining.sort("prob_yes", ascending=True).head(30)

most_likely.to_csv("most_likely_new_stands.csv", sep=",")
least_likely.to_csv("least_likely_new_stands.csv", sep=",")
