# AMULOG (A Manager for Unstructured LOGs)

## Overview

A tool to support system log management.  
The main function is to classify log messages with automatically generated log templates (formats and variable locations),
and to store the data in a database.  
This system works on python3.


## Main features

- Support multiple databases: sqlite and mysql
- Smart log segmentation with [log2seq](https://github.com/cpflat/log2seq)
- Multiple template generation algorithms like: Drain, SHISO, LenMa, FT-tree, Dlog, etc.
- Import/Export log templates
- Edit log templates manually
- Search API with datetime, hostname and log template IDs


## Tutorial

#### Install

<!--
git clone this repository, then try  
`$ python setup.py install`  
in the cloned directory.
-->

`$ pip install amulog`

#### Generate config

For the first step, save following config as `test.conf` on an empty directory.
```
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
```

Then modify `general.src_path` option to a logfile you want to load.  
(If you want to use multiple files, change `general.src_recur` into true and specify directory name to `general.src_path`.)

#### Generate database

Try following command to generate database:  
`$ python -m amulog db-make -c test.conf`


#### Check database
`python -m amulog show-db-info -c test.conf`  
shows status of the generated database.

`python -m amulog show-lt -c test.conf`  
shows all generated log templates in the given logfile.

`python -m amulog show-log -c test.conf ltid=2`  
shows all log messages corresponding to log template ID 2.


#### Resume generating database

Try following command to resume generating database:  
`$ python -m amulog db-add -c test.conf logfile2.txt`


#### Export and Import templates

Following command exports all log templates in the database:  
`python3 -m amulog show-db-import -c test.conf > exported_tpl.txt`

You can modify the exported templates manually.  
Note that some special letters (`\,@,*`) are escaped in the exported templates.  

To import the templates, save following config as `test2.conf`.
```
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
```

Then, try generating database again:  
`python -m amulog db-make -c test2.conf`

#### Further usage

see help with following command:  
`python -m amulog -h`


## Code
The source code is available at https://github.com/cpflat/amulog


## License

3-Clause BSD license


## Author

Satoru Kobayashi


