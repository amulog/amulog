#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule = [
    Date(),
    ItemGroup([Digit("hour"),
               Digit("minute"),
               Digit("second"),
               DemicalSecond()], separator=":,"),
    String("level"),
    UserItem("component", r".*"),
    Statement()
]
full_format = r"<0> <1> - <2>  \[<3>\] - <4>"

header_parser = HeaderParser(header_rule, full_format=full_format)

statement_parser = preset.default_statement_parser()

parser = LogParser(header_parser, statement_parser)

