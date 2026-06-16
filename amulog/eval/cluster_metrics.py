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
    # guard degenerate inputs (e.g. all-singleton clusters have no pairs, so
    # tp+fp == tp+fn == 0); undefined metrics are 0.0 (sklearn zero_division=0)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        fscore = (2. * precision * recall) / (precision + recall)
    else:
        fscore = 0.0
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
    """Ratio of ground-truth clusters that are over-divided.

    A ground-truth (answer) cluster is over-divided when its members are
    split across more than one estimated cluster. The ratio is that count
    divided by the number of ground-truth clusters.

    over-division / over-aggregation vs homogeneity (Ho) / completeness (Co):
        Both pairs describe the same two failure modes, but from different
        angles:

        - Ho/Co (V-measure support metrics; computed with sklearn) are
          *entropy-based and instance-weighted*: a large cluster mixing many
          log lines moves the score more than a small one. Use them for an
          overall, instance-level accuracy score (Co reflects over-division,
          Ho reflects over-aggregation).
        - These ratios are *cluster-counting and size-agnostic*: every
          cluster counts equally regardless of how many log lines it holds.
          They answer "what fraction of clusters (templates) are broken",
          which is a better proxy for the manual effort of fixing templates
          (one template is one unit of work irrespective of its size).

        They are complementary, not redundant, and generally yield different
        values. See also over_aggregation_cluster_ratio (the dual).
    """
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
    """Ratio of estimated clusters that are over-aggregated.

    This is the dual of over_division_cluster_ratio: an estimated cluster is
    over-aggregated when it contains members of more than one ground-truth
    cluster. The ratio is that count divided by the number of estimated
    clusters. See over_division_cluster_ratio for how these cluster-counting
    ratios complement the entropy-based Ho/Co metrics.

    Note:
        The earlier implementation inspected only the first estimated cluster
        per ground-truth cluster and was not invariant to cluster relabeling
        (the same clustering could yield different values depending on label
        names); it counted per ground-truth cluster, asymmetric with the dual
        above. This was a genuine defect, not a published metric (the paper
        measures over-division/over-aggregation via Ho/Co), so it was
        corrected to the label-invariant per-estimated-cluster definition.
    """
    from sklearn.metrics.cluster import contingency_matrix
    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    # denominator: number of estimated clusters (dual of over_division)
    n_cluster = np.unique(a_pred).shape[0]
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()

    n_fail_cluster = 0
    for uniq_label_pred, uniq_cnt_pred in zip(
            *np.unique(nz_pred, return_counts=True)):
        # 1 estimated cluster holding members of multiple answer clusters?
        if uniq_cnt_pred > 1:
            n_fail_cluster += 1

    return 1. * n_fail_cluster / n_cluster



