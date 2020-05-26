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
    """Parsing accuracy is one of clustering metrics.
    It is the ratio of log lines in completely-correct clusters.
    A completely-correct cluster has log instances (i.e., cluster members)
    that is completely same as the instances in an answer cluster.

    This metrics is defined in [1].
    [1] P. He, et al. Drain: An Online Log Parsing Approach with Fixed Depth Tree. ICWS 2017, pp.33–40, 2017.
    https://github.com/logpai/logparser is also referred.
    """

    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    n_total_line = a_true.shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()
    nz_cnt = cm.data

    n_correct_line = 0
    for uniq_label_true, uniq_cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        # 1 estimated cluster for 1 answer cluster?
        if uniq_cnt_true == 1:
            index_uniq_true = (nz_true == uniq_label_true)
            index_uniq_pred = (nz_pred == nz_pred[index_uniq_true][0])
            # 1 answer cluster for 1 estimated cluster?
            if nz_true[index_uniq_pred].shape[0] == 1:
                n_correct_line += sum(nz_cnt[index_uniq_true])

    return 1. * n_correct_line / n_total_line


def cluster_accuracy(labels_true, labels_pred):
    """Cluster accuracy is one of clustering metrics.
    This is an extended version of parsing accuracy,
    but counting clusters instead of log messages.
    """
    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    n_cluster = np.unique(a_true).shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()

    n_correct_cluster = 0
    for uniq_label_true, uniq_cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        # 1 estimated cluster for 1 answer cluster?
        if uniq_cnt_true == 1:
            index_uniq_true = (nz_true == uniq_label_true)
            index_uniq_pred = (nz_pred == nz_pred[index_uniq_true][0])
            # 1 answer cluster for 1 estimated cluster?
            if nz_true[index_uniq_pred].shape[0] == 1:
                n_correct_cluster += 1

    return 1. * n_correct_cluster / n_cluster


def over_division_cluster_ratio(labels_true, labels_pred):
    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    n_cluster = np.unique(a_true).shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()

    n_fail_cluster = 0
    for uniq_label_true, uniq_cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        # multiple estimated cluster for 1 answer cluster?
        if uniq_cnt_true > 1:
            n_fail_cluster += 1

    return 1. * n_fail_cluster / n_cluster


def over_aggregation_cluster_ratio(labels_true, labels_pred):
    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    n_cluster = np.unique(a_true).shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()

    n_fail_cluster = 0
    for uniq_label_true in np.unique(nz_true):
        index_uniq_true = (nz_true == uniq_label_true)
        index_uniq_pred = (nz_pred == nz_pred[index_uniq_true][0])
        # multiple answer cluster for 1 estimated cluster?
        if nz_true[index_uniq_pred].shape[0] > 1:
            n_fail_cluster += 1

    return 1. * n_fail_cluster / n_cluster



