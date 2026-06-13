#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import lt_common
from amulog.lt_search import LTSearchTree, LTSearchTreeNew

R = lt_common.REPLACER

# templates exercising shared prefixes, wildcard branches and varying lengths
_TEMPLATES = [
    (0, ["connect", "from", R, "port", R]),
    (1, ["connect", "to", R]),
    (2, [R, "from", R, "port", R]),
    (3, ["disconnect", R]),
    (4, ["connect", "from", "localhost", "port", "22"]),
]
_QUERIES = [
    ["connect", "from", "h1", "port", "22"],
    ["connect", "from", "localhost", "port", "22"],
    ["connect", "to", "server"],
    ["disconnect", "now"],
    ["x", "from", "h1", "port", "22"],
    ["no", "match", "here"],
    ["connect", "from", "h1", "port"],  # wrong length
]


def _build(cls):
    tree = cls()
    for ltid, tpl in _TEMPLATES:
        tree.add(ltid, tpl)
    return tree


class TestLTSearchTreeNew(unittest.TestCase):

    def test_remove_with_wildcard_backtrack(self):
        # A concrete word that has a wildcard sibling pushes a checkpoint;
        # when that word-branch dead-ends, _trace_path must backtrack onto
        # the *stacked* node's wildcard, not the current node's (CR-32).
        # Otherwise the stack-pop dereferences a node without a wildcard.
        tree = LTSearchTreeNew()
        tree.add(1, [R, "b"])
        tree.add(2, ["a", "c"])

        # search uses the (correct) _trace and resolves ["a","b"] to ltid 1
        self.assertEqual(tree.search(["a", "b"]), 1)

        # remove uses _trace_path and must resolve to the same node
        # (regression: raised KeyError "@wild")
        removed = tree.remove(["a", "b"])
        self.assertEqual(removed, 1)

        # the matched template is gone, the other one remains
        self.assertIsNone(tree.search(["a", "b"]))
        self.assertEqual(tree.search(["a", "c"]), 2)

    def test_remove_stored_template(self):
        # straightforward removal still works
        tree = LTSearchTreeNew()
        tree.add(1, ["a", "b", "c"])
        tree.add(2, ["a", R, "d"])
        self.assertEqual(tree.remove(["a", "b", "c"]), 1)
        self.assertIsNone(tree.search(["a", "b", "c"]))
        self.assertEqual(tree.search(["a", "x", "d"]), 2)


class TestLTSearchTreeEquivalence(unittest.TestCase):
    """LTSearchTree (legacy) and LTSearchTreeNew must behave identically.
    This keeps the legacy reference implementation honest and exercises it."""

    def test_search_equivalence(self):
        old = _build(LTSearchTree)
        new = _build(LTSearchTreeNew)
        for q in _QUERIES:
            self.assertEqual(old.search(q), new.search(q),
                             "search mismatch for {0}".format(q))

    def test_remove_nonstored_is_graceful(self):
        # regression (CR-69): legacy remove dereferenced a None trace and
        # raised AttributeError; it must no-op like the new implementation.
        old = _build(LTSearchTree)
        new = _build(LTSearchTreeNew)
        old.remove(99, ["nope", "not", "stored"])  # must not raise
        self.assertIsNone(new.remove(["nope", "not", "stored"]))
        # trees are unchanged: searches still agree
        for q in _QUERIES:
            self.assertEqual(old.search(q), new.search(q))

    def test_remove_equivalence(self):
        old = _build(LTSearchTree)
        new = _build(LTSearchTreeNew)
        target = ["connect", "to", R]
        old.remove(1, target)
        new.remove(target)
        for q in _QUERIES:
            self.assertEqual(old.search(q), new.search(q),
                             "post-remove search mismatch for {0}".format(q))


if __name__ == "__main__":
    unittest.main()
