#!/usr/bin/env python
# coding: utf-8

import builtins
import sys
import unittest

from amulog import lt_common

try:
    import amsemantics as _amsemantics
except ImportError:
    _amsemantics = None


class TestAmsemanticsOptional(unittest.TestCase):
    """amsemantics is an optional dependency: amulog.ltg_semantics must be
    importable without it, and selecting it must give an actionable error."""

    def test_importable_without_amsemantics(self):
        # ltg_semantics must import even if amsemantics is absent (no
        # top-level dependency on the optional package). Block amsemantics
        # and reload the module to prove it.
        import importlib
        from amulog import ltg_semantics
        real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == "amsemantics" or name.startswith("amsemantics."):
                raise ImportError("No module named 'amsemantics'")
            return real_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items()
                 if k == "amsemantics" or k.startswith("amsemantics.")}
        for k in saved:
            del sys.modules[k]
        builtins.__import__ = _block
        try:
            importlib.reload(ltg_semantics)  # must not raise
            self.assertTrue(hasattr(ltg_semantics, "LTGroupSemantics"))
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)
            importlib.reload(ltg_semantics)  # restore normal state

    def test_actionable_error_when_missing(self):
        from amulog import ltg_semantics
        real_import = builtins.__import__

        def _block_amsemantics(name, *args, **kwargs):
            if name == "amsemantics" or name.startswith("amsemantics."):
                raise ImportError("No module named 'amsemantics'")
            return real_import(name, *args, **kwargs)

        saved = {k: v for k, v in sys.modules.items()
                 if k == "amsemantics" or k.startswith("amsemantics.")}
        for k in saved:
            del sys.modules[k]
        builtins.__import__ = _block_amsemantics
        try:
            with self.assertRaises(ImportError) as cm:
                ltg_semantics._amsemantics()
            self.assertIn("amsemantics", str(cm.exception))
            self.assertIn("semantics", str(cm.exception))
        finally:
            builtins.__import__ = real_import
            sys.modules.update(saved)


@unittest.skipUnless(_amsemantics is not None, "amsemantics not installed")
class TestLTGroupSemantics(unittest.TestCase):

    def test_sc_kwargs_is_caller_snapshot(self):
        # CR-72: _sc_kwargs must snapshot the caller's kwargs, not the dict
        # mutated with the injected normalizer/training_sources/input_sources.
        from amulog import ltg_semantics
        group = ltg_semantics.LTGroupSemantics(
            lt_common.LTTable(), None,
            lda_knowledge_sources=[], lda_library="gensim")
        self.assertEqual(group._sc_kwargs, {"lda_library": "gensim"})
        for injected in ("normalizer", "training_sources", "input_sources"):
            self.assertNotIn(injected, group._sc_kwargs)

    def test_constructs_and_use_input_flag(self):
        from amulog import ltg_semantics
        group = ltg_semantics.LTGroupSemantics(
            lt_common.LTTable(), None, lda_knowledge_sources=["self"])
        self.assertTrue(group._use_input_as_training)
        self.assertIsNotNone(group.classifier)


if __name__ == "__main__":
    unittest.main()
