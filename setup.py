#!/usr/bin/env python

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
      version='0.0.2',
      description='',
      long_description=read_md('README.md'),
      author='Satoru Kobayashi',
      author_email='sat@nii.ac.jp',
      url='https://github.com/cpflat/amulog/',
      install_requires=['numpy>=1.9.2',
                        'scipy>=1.2.0',
                        'pandas>=1.0.0',
                        'scikit-learn>=0.20.2',
                        'python-dateutil>=2.8.0',
                        'log2seq>=0.0.3', ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.4.3',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=['amulog'],
      package_data={'amulog': data_files},
      test_suite="suite.suite"
      )
