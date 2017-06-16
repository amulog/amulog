#!/usr/bin/env python

from __future__ import print_function

import sys
import os
from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print('pandoc is not installed.')
    read_md = lambda f: open(f, 'r').read()

sys.path.append("./tests")
package_name = 'amulog'
data_dir = "/".join((package_name, "data"))
data_files = ["/".join((data_dir, fn)) for fn in os.listdir(data_dir)]

setup(name='amulog',
    version='0.0.1',
    description='',
    long_description=read_md('README.md'),
    author='Satoru Kobayashi',
    author_email='sat@hongo.wide.ad.jp',
    url='https://github.com/cpflat/amulog/',
    install_requires=['numpy>=1.9.2'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        ('License :: OSI Approved :: '
         'GNU General Public License v2 or later (GPLv2+)'),
        'Programming Language :: Python :: 3.4.3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    license='GNU General Public License v2 or later (GPLv2+)',
    
    packages=['amulog'],
    package_data={'amulog' : data_files},
    test_suite = "suite.suite"
    )
