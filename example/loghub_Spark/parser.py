#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule1 = [
    ItemGroup([YearWithoutCentury(),
               Digit("month"),
               Digit("day")], separator="/"),
    Time(),
    String("level"),
    UserItem("component", r"[a-zA-Z0-9.]+"),
    Statement()
]

header_rule2 = [
    Statement()
]

header_parser1 = HeaderParser(header_rule1, separator=" :")
header_parser2 = HeaderParser(header_rule2, reformat_timestamp=False)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)

