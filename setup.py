#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup


def load_readme():
    with open('README.md', 'r') as f:
        return f.read()


sys.path.append("./tests")
package_name = 'amulog'
data_dir = "/".join((package_name, "data"))
data_files = ["/".join((data_dir, fn)) for fn in os.listdir(data_dir)]

init_path = os.path.join(os.path.dirname(__file__), package_name, '__init__.py')
with open(init_path) as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

setup(name=package_name,
      version=version,
      description='A system log management tool with automatically generated log templates.',
      long_description=load_readme(),
      long_description_content_type='text/markdown',
      author='Satoru Kobayashi',
      author_email='sat@nii.ac.jp',
      url='https://github.com/cpflat/amulog/',
      install_requires=['numpy>=1.9.2',
                        'scipy>=1.2.0',
                        'scikit-learn>=0.20.2',
                        'python-dateutil>=2.8.0',
                        'log2seq>=0.1.4', ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=['amulog'],
      package_data={'amulog': data_files},
      test_suite="tests"
      )
