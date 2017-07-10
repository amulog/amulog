#!/usr/bin/env python
# coding: utf-8

import os

import pycrfsuite


class FeatureExtracter():
    DEFAULT_TEMPLATE = "/".join((os.path.dirname(__file__),
                                 "../data/crf_template"))
    TPL_FIELD = {"0": 0, "w": 0, "word": 0,
                 "1": 1, "pos": 1, "mid": 1,
                 "2": 2, "y": 2, "label": 2,
                 "bos": None, "eos": None,
                 "F": None}

    def __init__(self, template_fp = None):
        if template_fp is None or template_fp == "":
            self.template = self.load_template(self.DEFAULT_TEMPLATE)
        else:
            self.template = self.load_template(template_fp)
        self.bos = False
        self.eos = False

    @classmethod
    def _tpl_field(cls, field, offset):
        if field in cls.TPL_FIELD:
            val = cls.TPL_FIELD[field]
            if val is None:
                if field == "bos":
                    self.bos = True
                elif field == "eos":
                    self.eos = True
                else:
                    raise NotImplementedError(
                       "Invalid syntax of crf feature template")
                return None
            else:
                return val
        else:
            raise NotImplementedError(
                "Invalid syntax of crf feature template")

    def load_template(self, fp):
        template = [] # (field (int), offset (int))
        with open(fp, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                l_cond = [s.strip() for s in line.split(",")]
                name = "|".join(l_cond)
                l_rule = []
                for cond in l_cond:
                    field, n, offset = cond.rstrip("]").partition("[")
                    field_id = self._tpl_field(field, offset)
                    if field_id is not None:
                        l_rule.append([field, int(offset)])
                template.append([name, l_rule])
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
            for name, l_rule in self.template:
                subfeature = []
                for field, offset in l_rule:
                    p = wid + offset
                    if p in l_items_range:
                        temp_item = l_items[p]
                        val = self._get_item(temp_item, field)
                        subfeature.append(val)
                fval = "|".join(subfeature)
                d_feature[name] = fval
            ret.append(d_feature)

        if self.bos:
            ret[0]["F"] = "__BOS__"
        if self.eos:
            ret[-1]["F"] = "__EOS__"

        return pycrfsuite.ItemSequence(ret)

    def label(self, l_items):
        return [item[-1] for item in l_items]


