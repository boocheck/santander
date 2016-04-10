#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import numpy as np
import santanderenv as senv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as pps
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.metrics as metrics
from sklearn import cross_validation
import scipy as sp
from datetime import datetime
from sklearn.neural_network import BernoulliRBM
import featsprep as fp



def dump_to_file(filename, ids, targets):
    np.savetxt(os.path.join(senv.get_config("results"), filename + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S") + ".txt"),
               np.concatenate((ids, targets), 1), header="ID,TARGET", fmt="%d,%f", comments="")


def read_headers(filename, delitemer=","):
    file = open(filename)
    line = file.readline()
    file.close()
    return line.split(delitemer)


def find_all(collection, items_to_find):
    return [x[0] for x in filter(lambda it: it[1] in items_to_find, enumerate(collection))]


preserved_feats = [
    "saldo_var30",
    "var15",
    "saldo_medio_var5_hace2",
    "saldo_var42",
    "num_var4",
    "num_var35",
    "saldo_medio_var5_ult1",
    "saldo_medio_var5_ult3",
    "saldo_var5",
    "num_meses_var5_ult3",
    "saldo_medio_var5_hace3",
    "num_var30",
    "ind_var30",
    "num_var42",
    "num_var5",
    "ind_var5"
]

names = [
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
    "Linear Discriminant Analysis",
    "Quadratic Discriminant Analysis",
    "Linear SVM",
    "RBF SVM",
    "Nearest Neighbors"
]

classifiers = [
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=100),  #(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    KNeighborsClassifier(3)
]

headers = read_headers(senv.get_config("data-test"))


# TODO FIX THIS LINES
train_all = np.loadtxt(senv.get_config("data-train"), skiprows=1, delimiter=",")
test_all = np.loadtxt(senv.get_config("data-test"), skiprows=1, delimiter=",")
# train_all[:3, -1:] = 1
ids = test_all[:, [0]]
train_len = train_all.shape[0]


# preprocess dataset, split into training and test part
preserved_indices = find_all(headers, preserved_feats)
all_feats = pps.MinMaxScaler().fit_transform(np.concatenate((train_all[:, 1:-1], test_all[:, 1:])))
X_train = all_feats[:train_len, :]
X_test = all_feats[train_len:, :]
X_train_preserved = all_feats[:train_len, preserved_indices]
X_test_preserved = all_feats[train_len:, preserved_indices]
#fp.review_data(np.concatenate((X_train, X_test)))
y_train = train_all[:, -1]
# y_test = test_all[:, -1]

# permutate
rs = np.random.RandomState(0)
train_permut = rs.permutation(X_train.shape[0])
#test_permut = rs.permutation(X_test.shape[0])
X_train = X_train[train_permut, :]
X_train_preserved = X_train_preserved[train_permut, :]
y_train = y_train[train_permut]


# X_test = X_test[test_permut, :]
# y_test = y_test[test_permut]


# rbm learning
# TODO: try to search better parametrs with grid search
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm.learning_rate = 0.1
rbm.n_iter = 30
rbm.n_components = 16

print X_train
print X_train.shape
rbm.fit(all_feats)
X_train = np.concatenate((rbm.transform(X_train), X_train_preserved), 1)
X_test = np.concatenate((rbm.transform(X_test), X_test_preserved), 1)
print X_train
print X_train.shape


ens_lbls = []
ens_probs = []
# iterate over classifiers
for name, clf in zip(names, classifiers):
    print "[{}] learning starting ...".format(name)
    clf.fit(X_train, y_train)
    print "[{}] learning finished".format(name)
    probs = clf.predict_proba(X_test)[:, [1]]
    dump_to_file(name+"_res_probs", ids, probs)

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
    print name, scores.tolist(), np.mean(scores)


    # lbls = clf.predict(X_test)


    # acc = metrics.accuracy_score(y_test, lbls)
    # auc = metrics.roc_auc_score(y_test, probs)

    # print "[{}] acc: {}   auc: {}".format(name, acc, auc)
    # np.savetxt(name+"_res_lbls.txt", lbls)


    # if(lbls.ndim==1):
    #     ens_lbls.append(np.reshape(lbls, [-1, 1]))
    # else:
    #     ens_lbls.append(lbls)
    # ens_probs.append(probs)




# concd_lbls = np.concatenate(ens_lbls, 1)
# concd_probs = np.concatenate(ens_probs, 1)
#
#
# dump_to_file("mode", ids, sp.stats.mode(concd_lbls, 1)[0])
# dump_to_file("mean", ids, np.mean(concd_probs, 1, keepdims=True))
# dump_to_file("median", ids, np.median(concd_probs, 1, keepdims=True))




