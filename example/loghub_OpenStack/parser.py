#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule1 = [
    UserItem("", r"[^ ]+", dummy=True),
    Date(),
    Time(),
    Digit("processid"),
    String("level"),
    Hostname("host"),
    UserItem("addr", r"[a-zA-Z0-9 -]+"),
    Statement()
]

header_rule2 = [
    Statement()
]

full_format=r"<0> <1> <2> <3> <4> <5> \[<6>\] <7>"

header_parser1 = HeaderParser(header_rule1, full_format=full_format)
header_parser2 = HeaderParser(header_rule2, reformat_timestamp=False)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)

