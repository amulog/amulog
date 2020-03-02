#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import combinations
from scipy.special import comb

X = [0,0,1,2,2,2,2,3,4,4] + [4,4,4,4]
Y = [0,0,1,1,2,2,3,3,4,4] + [5,5,5,1]


def conf_matrix1(x, y):
    n_pair = 0
    tp = 0; fp = 0; fn = 0; tn = 0
    for pair_answer, pair_trial in zip(combinations(x, 2),
                                       combinations(y, 2)):
        n_pair += 1
        if pair_answer[0] == pair_answer[1]:
            if pair_trial[0] == pair_trial[1]:  # True Positive
                tp += 1
            else:  # False Negative
                fn += 1
        else:
            if pair_trial[0] == pair_trial[1]:  # False Positive
                fp += 1
            else:  # True Negative
                tn += 1

    print(n_pair)
    return tp, fp, fn, tn


def conf_matrix2(x, y):
    from amulog.eval import cluster_metrics
    tp, fp, fn, tn = cluster_metrics.confusion_matrix(x, y).ravel()
    return tp, fp, fn, tn


print(conf_matrix1(X, Y))
print(conf_matrix2(X, Y))




#def ri1(x, y):
#    n_pair = 0
#    tpfn = 0
#    for pair_answer, pair_trial in zip(combinations(x, 2),
#                                       combinations(y, 2)):
#        n_pair += 1
#        if (pair_answer[0] == pair_answer[1]) == (pair_trial[0] == pair_trial[1]):
#            tpfn += 1
#    print("{0} / {1}".format(tpfn, n_pair))
#    return tpfn / n_pair
#
#
#def ri2(x, y):
#    x = np.array(x)
#    m_x_diff = np.abs(x - x[:, np.newaxis])
#    m_x_diff[m_x_diff > 1] = 1
#    y = np.array(y)
#    m_y_diff = np.abs(y - y[:, np.newaxis])
#    m_y_diff[m_y_diff > 1] = 1
#    sg = np.sum(np.abs(m_x_diff - m_y_diff)) / 2
#    all_comb = comb(len(x), 2, exact=True)
#    print("sg: {0}, all_comb: {1}".format(sg, all_comb))
#    return 1 - sg / all_comb
#
#print("start")
#print(ri1(X, Y))
#print(ri2(X, Y))
