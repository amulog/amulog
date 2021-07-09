import os
import re
import logging
import configparser
from abc import ABC, abstractmethod

_logger = logging.getLogger(__package__)
# DEFAULT_LABEL_CONF = "/".join((os.path.dirname(__file__),
#                                "data/lt_label.conf.sample"))


class LTTagger(ABC):

    def __init__(self):
        pass

    def get_tags(self, ltobj):
        raise NotImplementedError


class LTLabelDummy(LTTagger):

    def get_tags(self, _):
        return []


class LTLabelINI(LTTagger):
    """
    Note:
        Configuration defined earlier is prior.
    """

    group_header = "group_"
    label_header = "label_"

    def __init__(self, conf_fn, tag="group", default_label=None, default_group=None):
        from . import config
        super().__init__()
        if not os.path.exists(conf_fn):
            raise IOError("{0} not found".format(conf_fn))
        self.conf = configparser.ConfigParser()
        self.conf.read(conf_fn)
        self._tag = tag
        self.default_label = default_label
        self.default_group = default_group

        self.groups = []
        self.labels = []
        for sec in self.conf.sections():
            if sec[:len(self.group_header)] == self.group_header:
                self.groups.append(sec[len(self.group_header):])
            elif sec[:len(self.label_header)] == self.label_header:
                self.labels.append(sec[len(self.label_header):])

        self.d_group = {}  # key : group, val : [label, ...]
        self.d_rgroup = {}  # key : label, val : [group, ...]
        for group in self.groups:
            section = self.group_header + group
            for label in config.gettuple(self.conf, section, "members"):
                self.d_group.setdefault(group, []).append(label)
                self.d_rgroup.setdefault(label, []).append(group)
        self.rules = []  # [(label, (re_matchobj, ...)), ...]
        for label in self.labels:
            section = self.label_header + label
            for rulename in config.gettuple(self.conf, section, "rules"):
                l_re = []
                l_restr = config.gettuple(self.conf, section, rulename)
                for re_str in l_restr:
                    if rulename[0] == "i":
                        re_obj = re.compile(re_str, re.IGNORECASE)
                    elif rulename[0] == "c":
                        re_obj = re.compile(re_str, )
                    else:
                        raise ValueError(
                                "lt_label rulename invalid ({0})".format(
                                    rulename))
                    l_re.append(re_obj)
                self.rules.append((label, tuple(l_re)))

    @staticmethod
    def _test_rule(ltobj, t_re):
        for reobj in t_re:
            for word in ltobj.ltw:
                if reobj.match(word):
                    break
                else:
                    pass
            else:
                # no word matches
                return False
        else:
            # all reobj passes
            return True

    def get_tags(self, ltobj):
        if self._tag == "group":
            return [self.get_lt_group(ltobj)]
        elif self._tag == "label":
            return [self.get_lt_label(ltobj)]
        elif self._tag == "all":
            group = self.group_header + self.get_lt_group(ltobj)
            label = self.label_header + self.get_lt_label(ltobj)
            return [group, label]
        else:
            raise ValueError

    def get_lt_label(self, ltline):
        for label, t_re in self.rules:
            if self._test_rule(ltline, t_re):
                return label
        else:
            return self.default_label

    def get_lt_group(self, ltline):
        label = self.get_lt_label(ltline)
        return self.get_group(label)

    def get_group(self, label):
        if label is None:
            return self.default_group
        else:
            if label in self.d_rgroup:
                group = self.d_rgroup[label]
                assert len(group) == 1
                return group[0]
            else:
                return self.default_group


def init_ltlabeldummy(_):
    return LTLabelDummy()


def init_ltlabelini(conf):
    tag_file = conf.get("visual", "tag_file")
    tag_key = conf.get("visual", "tag_file_key")
    default_label = conf.get("visual", "tag_file_default_label")
    default_group = conf.get("visual", "tag_file_default_group")
    return LTLabelINI(tag_file, tag_key,
                      default_label=default_label,
                      default_group=default_group)


def generate_all_tags(conf, reset=True):
    tag_method = conf.get("visual", "tag_method")
    if tag_method == "dummy":
        tagger = init_ltlabeldummy(conf)
    elif tag_method in ("file", "ini"):
        tagger = init_ltlabelini(conf)
    else:
        raise ValueError("Invalid visual.tag_method")

    from . import log_db
    ld = log_db.LogData(conf, edit=True)
    if reset:
        ld.db.reset_tag()
    for ltobj in ld.iter_lt():
        tags = tagger.get_tags(ltobj)
        ld.db.add_tags(ltobj.ltid, tags)
