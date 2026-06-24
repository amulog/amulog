#!/usr/bin/env python
# coding: utf-8

"""Regression tests pinning the LTGen algorithm classification that
``amulog show-algorithms`` derives from code.

The Wiki previously hand-maintained a stateful/stateless table and got
several entries wrong (drain/shiso/lenma mislabelled as stateless, and
parallel-safety inverted). These tests fix the correct classification as
machine-checked expectations so future refactors cannot silently drift,
and so the easy-to-confuse cases (offline != stateless; LTGenOffline is
stateful) stay honest.
"""

import unittest

from amulog import config
from amulog import manager
from amulog import lt_common


# name -> (classname, stateful, processing, parallel)
# Truth source: amulog/lt_common.py class hierarchy (is_stateful) and the
# online/offline registration lists in amulog/alg/meta.py.
EXPECTED = {
    "shiso":      ("LTGenSHISO",             True,  "online",  False),
    "drain":      ("LTGenDrain",             True,  "online",  False),
    "lenma":      ("LTGenLenMa",             True,  "online",  False),
    "fttree":     ("LTGenFTTree",            True,  "online",  False),
    "va":         ("LTGenVA",                True,  "offline", False),
    "dlog":       ("LTGenDlog",              True,  "offline", False),
    "import":     ("LTGenImport",            False, "any",     True),
    "import-ext": ("LTGenImportExternal",    False, "any",     True),
    "re":         ("LTGenRegularExpression", False, "any",     True),
    "crf":        ("LTGenCRF",               False, "any",     True),
}


class TestShowAlgorithms(unittest.TestCase):

    def test_known_classification(self):
        info = {d["name"]: d for d in manager.get_algorithm_info()}
        self.assertEqual(set(info), set(EXPECTED))
        for name, exp in EXPECTED.items():
            classname, stateful, processing, parallel = exp
            d = info[name]
            self.assertEqual(d["classname"], classname, name)
            self.assertEqual(d["stateful"], stateful, name)
            self.assertEqual(d["processing"], processing, name)
            self.assertEqual(d["parallel"], parallel, name)

    def test_parallel_equals_stateless(self):
        # parallel (offline multiprocessing) is allowed exactly for
        # stateless generators (see LTGenStateless docstring).
        for d in manager.get_algorithm_info():
            self.assertEqual(d["parallel"], not d["stateful"], d["name"])

    def test_offline_is_not_implied_stateless(self):
        # Guards the classic mistake "offline => stateless". Both offline
        # algorithms remain stateful and not parallel-safe.
        info = {d["name"]: d for d in manager.get_algorithm_info()}
        for name in ("va", "dlog"):
            self.assertEqual(info[name]["processing"], "offline", name)
            self.assertTrue(info[name]["stateful"], name)
            self.assertFalse(info[name]["parallel"], name)

    def test_get_ltgen_class_returns_ltgen_subclass(self):
        for name in EXPECTED:
            cls = manager.get_ltgen_class(name)
            self.assertTrue(issubclass(cls, lt_common.LTGen), name)

    def test_derived_stateful_matches_instance(self):
        # The class-based derivation in get_algorithm_info() must agree
        # with the actual is_stateful() of a built instance. This catches
        # a future algorithm whose is_stateful() override diverges from
        # its base class. Limited to algorithms that build from default
        # config (no model/definition file required).
        info = {d["name"]: d for d in manager.get_algorithm_info()}
        for name in ("shiso", "drain", "lenma", "fttree", "va", "dlog"):
            conf = config.open_config(verbose=False)
            conf["log_template"]["lt_methods"] = name
            table = lt_common.TemplateTable()
            ltgen = manager.init_ltgen_methods(conf, table)
            self.assertEqual(info[name]["stateful"],
                             ltgen.is_stateful(), name)


if __name__ == "__main__":
    unittest.main()
