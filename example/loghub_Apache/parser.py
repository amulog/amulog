#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule1 = [
    ItemGroup([String("weekday", dummy=True),
               MonthAbbreviation(),
               Digit("day"),
               Time(),
               Digit("year")],
              separator=" "),
    String("severityname"),
    Statement()
]
# The brackets around time and level are fixed delimiters, so pin them with
# full_format instead of treating "[" / "]" as generic separators. This keeps a
# leading "[client <ip>]" (part of loghub's <Content>) inside the message rather
# than consuming it as a host item.
full_format = r"\[<0>\] \[<1>\] <2>"
header_parser1 = HeaderParser(header_rule1, full_format=full_format)

header_rule2 = [
    Statement()
]
header_parser2 = HeaderParser(header_rule2, reformat_timestamp=False)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)
