#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup, find_packages


def load_readme():
    with open('README.rst', 'r') as fd:
        return fd.read()


def load_requirements():
    """Parse requirements.txt"""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as fd:
        requirements = [line.rstrip() for line in fd]
    return requirements


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
      author='Satoru Kobayashi',
      author_email='sat@3at.work',
      url='https://github.com/amulog/amulog/',
      python_requires='>=3.8',
      install_requires=load_requirements(),
      # optional features whose dependencies are published on PyPI; install
      # with e.g. `pip install amulog[crf]`. lt_methods=crf needs
      # python-crfsuite, ltgroup_alg=ssdeep needs ssdeep, database=mysql needs
      # pymysql. (The semantics ltgroup relies on packages that are not
      # published, so it is intentionally not listed here.)
      extras_require={
          "crf": ["python-crfsuite"],
          "ssdeep": ["ssdeep"],
          "mysql": ["pymysql"],
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          "Intended Audience :: Developers",
          'License :: OSI Approved :: BSD License',
          "Operating System :: OS Independent",
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Programming Language :: Python :: 3.14',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=find_packages(),
      package_data={'amulog': data_files},
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'amulog = amulog.__main__:main',
              'amulog.edit = amulog.edit.__main__:main',
              'amulog.eval = amulog.eval.__main__:main',
          ],
      },
      test_suite="tests"
      )
