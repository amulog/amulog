#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import numpy as np

from amulog.lt_common import REPLACER


def word_accuracy(tpls_true, tpls_pred):
    n_word = 0
    n_correct_word = 0
    for tpl_true, tpl_pred in zip(tpls_true, tpls_pred):
        for word_true, word_pred in zip(tpl_true, tpl_pred):
            n_word += 1
            if word_true == word_pred:
                n_correct_word += 1
    return 1. * n_correct_word / n_word


def line_accuracy(tpls_true, tpls_pred):
    n_line = 0
    n_correct_line = 0
    for tpl_true, tpl_pred in zip(tpls_true, tpls_pred):
        n_line += 1
        if tpl_true == tpl_pred:
            n_correct_line += 1
    return 1. * n_correct_line / n_line


def tpl_accuracy(tpls_true, tpls_pred, labels_true):
    d_n_line = defaultdict(int)
    d_n_correct_line = defaultdict(int)
    for tpl_true, tpl_pred, cluster in zip(tpls_true, tpls_pred,
                                           labels_true):
        d_n_line[cluster] += 1
        if tpl_true == tpl_pred:
            d_n_correct_line[cluster] += 1

    l_line_accuracy = [1. * d_n_correct_line[label] / d_n_line[label]
                       for label in d_n_line]
    return np.average(l_line_accuracy)


def tpl_word_accuracy(tpls_true, tpls_pred, labels_true):
    d_n_word = defaultdict(int)
    d_n_correct_word = defaultdict(int)
    for tpl_true, tpl_pred, cluster in zip(tpls_true, tpls_pred,
                                           labels_true):
        for word_true, word_pred in zip(tpl_true, tpl_pred):
            d_n_word[cluster] += 1
            if word_true == word_pred:
                d_n_correct_word[cluster] += 1
    l_word_accuracy = [1. * d_n_correct_word[label] / d_n_word[label]
                       for label in d_n_word]
    return np.average(l_word_accuracy)


def tpl_desc_accuracy(tpls_true, tpls_pred, labels_true):
    d_n_desc = defaultdict(int)
    d_n_correct_desc = defaultdict(int)
    for tpl_true, tpl_pred, cluster in zip(tpls_true, tpls_pred,
                                           labels_true):
        for word_true, word_pred in zip(tpl_true, tpl_pred):
            if word_true != REPLACER:
                d_n_desc[cluster] += 1
                if word_true == word_pred:
                    d_n_correct_desc[cluster] += 1
    l_ratio = [1. * d_n_correct_desc[label] / d_n_desc[label]
               for label in d_n_desc]
    return np.average(l_ratio)


def tpl_var_accuracy(tpls_true, tpls_pred, labels_true):
    d_n_var = defaultdict(int)
    d_n_correct_var = defaultdict(int)
    for tpl_true, tpl_pred, cluster in zip(tpls_true, tpls_pred,
                                           labels_true):
        for word_true, word_pred in zip(tpl_true, tpl_pred):
            if word_true == REPLACER:
                d_n_var[cluster] += 1
                if word_true == word_pred:
                    d_n_correct_var[cluster] += 1
    l_ratio = [1. * d_n_correct_var[label] / d_n_var[label]
               for label in d_n_var]
    return np.average(l_ratio)
