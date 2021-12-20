########################################
AMULOG (A Manager for Unstructured LOGs)
########################################

.. image:: https://img.shields.io/pypi/v/amulog
   :alt: PyPI release
   :target: https://pypi.org/project/amulog/

.. image:: https://img.shields.io/pypi/pyversions/amulog
   :alt: Python support
   :target: https://pypi.org/project/amulog/

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :alt: BSD 3-Clause License
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://api.travis-ci.com/amulog/amulog.svg?branch=master
   :alt: Travis CI
   :target: https://travis-ci.com/github/amulog/amulog


Amulog is a tool to support system log management.
The main function is to classify log messages with automatically generated log templates (formats and variable locations),
and to store the data in a database.
This system works on python3.

* Source: https://github.com/amulog/amulog
* Bug Reports: https://github.com/amulog/amulog/issues
* Author: `Satoru Kobayashi <https://github.com/cpflat/>`_
* License: `BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`_


Main features
=============

- Support multiple databases: sqlite and mysql
- Smart log segmentation with `log2seq <https://github.com/amulog/log2seq>`_
- Multiple template generation algorithms such as: Drain, SHISO, LenMa, FT-tree, Dlog, etc.
- Support Online (incremental) and Offline (hindsight) use
- Suspend and resume the template generation process
- Import and Export log templates if you need
- Edit log templates manually if you need
- Search API with datetime, hostname and log template IDs


Tutorial
========

Install
-------

::

    $ pip install amulog


Generate config
---------------

For the first step, save following config as :code:`test.conf` on an empty directory.

::

    [general]
    src_path = logfile.txt
    src_recur = false
    logging = auto.log

    [database]
    database = sqlite3
    sqlite3_filename = log.db

    [log_template]
    lt_methods = drain
    indata_filename = ltgen.dump

Then modify :code:`general.src_path` option to a logfile you want to load.
(If you want to use multiple files, change :code:`general.src_recur` into true and specify directory name to :code:`general.src_path`.)


Generate database
-----------------

Try following command to generate database:

::

    $ python -m amulog db-make -c test.conf


Check database
--------------

::

    $ python -m amulog show-db-info -c test.conf

shows status of the generated database.

::

    $ python -m amulog show-lt -c test.conf

shows all generated log templates in the given logfile.

::

    $ python -m amulog show-log -c test.conf ltid=2

shows all log messages corresponding to log template ID 2.


Resume generating database
--------------------------

Try following command to resume generating database:

::

    $ python -m amulog db-add -c test.conf logfile2.txt


Export and Import templates
---------------------------

Following command exports all log templates in the database:

::

    $ python3 -m amulog show-db-import -c test.conf > exported_tpl.txt

You can modify the exported templates manually.
Note that some special letters (:code:`\\`, :code:`@`, :code:`*`) are escaped in the exported templates.

To import the templates, save following config as :code:`test2.conf`.

::

    [general]
    src_path = logfile.txt
    src_recur = false
    logging = new_auto.log

    [database]
    database = sqlite3
    sqlite3_filename = new_log.db

    [log_template]
    lt_methods = import
    indata_filename = new_ltgen.dump

    [log_template_import]
    def_path = exported_tpl.txt

Then, try generating database again:

::

    python -m amulog db-make -c test2.conf


Further usage
-------------

see help with following command:

::

    python -m amulog -h


Reference
=========

This tool is demonstrated at `CNSM2020 <http://dl.ifip.org/db/conf/cnsm/cnsm2020/>`_ and `International Journal of Network Management <http://doi.org/10.1002/nem.2195>`_ (now on early access).

If you use this code, please consider citing:

::

    @inproceedings{Kobayashi_CNSM2020,
      author = {Kobayashi, Satoru and Yamashiro, Yuya and Otomo, Kazuki and Fukuda, Kensuke},
      booktitle = {Proceedings of the 16th International Conference on Network and Service Management (CNSM'20)},
      title = {amulog: A General Log Analysis Framework for Diverse Template Generation Methods},
      pages={1-5},
      year = {2020}
    }
