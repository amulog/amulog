#!/usr/bin/env python
# coding: utf-8

"""Convert Sequencial-string training data into ItemSequence data
by feature generation based on feature definitions.
The features are defined by 3 components.
- crf feature template: defining positional relations of each features
- crf label definition: defining mid-labels for feature "pos"
- input data (Sequencial-string)
"""

import os
import re

import pycrfsuite


class FeatureExtracter:
    DEFAULT_TEMPLATE = "/".join((os.path.dirname(os.path.abspath(__file__)),
                                 "../../data/crf_template"))
    TPL_PARSER = re.compile(r"^([a-z0-9]+)(\[[-0-9]+\])?$")
    TPL_FIELD = {"0": 0, "w": 0, "word": 0,
                 "1": 1, "pos": 1, "mid": 1,
                 "2": 2, "y": 2, "label": 2,
                 "bos": None, "eos": None,
                 "F": None}

    def __init__(self, template_fp=None, ig_key=[], ig_weight=0.1):
        if template_fp is None or template_fp == "":
            self._template = self._load_template(self.DEFAULT_TEMPLATE)
        else:
            self._template = self._load_template(template_fp)
        self._ig_key = ig_key  # unknown keys
        self._ig_val = ig_weight  # weight for unknown keys
        self._bos = False
        self._eos = False

    def _load_template_line(self, line):
        l_rule = []
        line = line.rstrip()
        if line == "":
            return None

        if ":" in line:
            temp = line.split(":")
            line = temp[0].strip()
            weight = float(temp[1].strip())
        else:
            weight = 1.0
        l_cond = [s.strip() for s in line.split(",")]
        name = "|".join(l_cond)
        for cond in l_cond:
            m = self.TPL_PARSER.match(cond)
            if m is None:
                raise SyntaxError(
                    "Invalid syntax of feature template ({0})".format(
                        line))
            field, offset_str = m.groups()
            if offset_str is None:
                offset = None
            else:
                offset = int(offset_str.strip("[").rstrip("]"))
            l_rule.append(tuple([field, offset]))
        return name, tuple(l_rule), weight

    def _load_template(self, fp):
        template = []  # name, tuples(field (int), offset (int))), weight
        with open(fp, 'r') as f:
            for line in f:
                ret = self._load_template_line(line)
                if ret is not None:
                    template.append(tuple(ret))
        return tuple(template)

    def _get_item(self, item, field):
        if field in self.TPL_FIELD:
            return item[self.TPL_FIELD[field]]
        else:
            raise NotImplementedError

    def feature(self, l_items):
        l_items_range = range(len(l_items))
        ret = []
        for wid, item in enumerate(l_items):
            d_feature = {}
            for name, l_rule, weight in self._template:
                subfeature = []
                for field, offset in l_rule:
                    if field == "bos":
                        if wid == 0:
                            subfeature.append("__BOS__")
                        else:
                            subfeature = None
                            break
                    elif field == "eos":
                        if wid == len(l_items) - 1:
                            subfeature.append("__EOS__")
                        else:
                            subfeature = None
                            break
                    else:
                        p = wid + offset
                        if p in l_items_range:
                            temp_item = l_items[p]
                            val = self._get_item(temp_item, field)
                            subfeature.append(val)
                        else:
                            subfeature = None
                            break
                if subfeature is not None:
                    s = set(subfeature)
                    if len(s) == 1:
                        if len(s & set(self._ig_key)) == 1:
                            weight = weight * self._ig_val
                    fval = "|".join(subfeature)
                    key = "=".join((name, fval))
                    d_feature[key] = weight
            ret.append(d_feature)

        return pycrfsuite.ItemSequence(ret)

    @staticmethod
    def label(l_items):
        return [item[-1] for item in l_items]
