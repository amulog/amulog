#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


# This example ignores some broken lines (in full HPC dataset)
# with too small value for unixtime.


header_rule = [
    Digit("logid"),
    String("class", optional=True),
    UserItem("node", r"[a-zA-Z0-9-]+", optional=True),
    UserItem("component", r"[a-zA-Z._-]+"),
    UserItem("state", r"[a-zA-Z._-]+"),
    UnixTime(),
    UserItem("flag", r"[0-9-]+"),
    Statement()
]

header_parser = HeaderParser(header_rule, reformat_timestamp=False)

statement_parser = preset.default_statement_parser()

parser = LogParser(header_parser, statement_parser, ignore_failure=True)

