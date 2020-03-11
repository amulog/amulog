#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import comb


def _comb2(n):
    return comb(n, 2, exact=True)


def confusion_matrix(labels_true, labels_pred):
    from sklearn.metrics.cluster import contingency_matrix

    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()
    nz_cnt = cm.data

    tp = sum(_comb2(n_ij) for n_ij in cm.data)

    # false negative: same cluster in labels_true, but different in labels_pred
    fn = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_true == uniq_label]
            # add combinations except true_positive part
            fn += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    # false positive: different cluster in labels_true, but same in labels_pred
    fp = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_pred, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_pred == uniq_label]
            # add combinations except true_positive part
            fp += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    total = _comb2(a_true.shape[0])
    tn = total - tp - fn - fp

    return np.array([[tp, fp], [fn, tn]])


def rand_score(labels_true, labels_pred):
    tp, fp, fn, tn = confusion_matrix(labels_true, labels_pred).ravel()
    return (tp + tn) / (tp + fp + fn + tn)


def precision_recall_fscore(labels_true, labels_pred):
    tp, fp, fn, tn = confusion_matrix(labels_true, labels_pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2. * precision * recall) / (precision + recall)
    return precision, recall, fscore


def parsing_accuracy(labels_true, labels_pred):
    # Referred https://github.com/logpai/logparser

    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    n_total = a_true.shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, _ = cm.nonzero()
    nz_cnt = cm.data

    n_correct = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt == 1:
            n_correct += sum(nz_cnt[nz_true == uniq_label])

    return 1. * n_correct / n_total


def cluster_accuracy(labels_true, labels_pred):
    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, _ = cm.nonzero()

    n_correct = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt == 1:
            n_correct += 1

    return 1. * n_correct / np.unique(a_true).shape[0]

