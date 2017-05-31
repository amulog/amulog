#!/usr/bin/env python
# coding: utf-8

import unittest


def suite():
    test_suite = unittest.TestSuite()
    l_suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    for ts in l_suite:
        test_suite.addTest(ts)
    return test_suite


if __name__ == "__main__":
    s = suite()
    unittest.TextTestRunner().run(s)

