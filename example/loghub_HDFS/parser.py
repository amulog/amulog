#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


# note: For HDFS_2 logs, use the parser script
# for Hadoop dataset (not for HDFS).


header_rule = [
    DateConcat(no_century=True),
    TimeConcat(),
    Digit("processid"),
    String("level"),
    UserItem("component", r"[a-zA-Z0-9.$]+"),
    Statement()
]

header_parser = HeaderParser(header_rule, separator=" :")

statement_parser = preset.default_statement_parser()

parser = LogParser(header_parser, statement_parser)

