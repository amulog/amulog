#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule = [
    String("label", symbols="-"),
    Digit("unixtime", dummy=True),
    ItemGroup([Digit("year", dummy=True),
               Digit("month", dummy=True),
               Digit("day", dummy=True)],
               separator="."),
    String("host", symbols=":_-", dummy=True),
    ItemGroup([Digit("year"),
               Digit("month"),
               Digit("day"),
               Digit("hour"),
               Digit("minute"),
               Digit("second"),
               DemicalSecond("dsecond")],
               separator="-."),
    String("host", symbols=":_-"),
    String("type"),
    String("component", symbols="_"),
    String("level"),
    Statement(optional=True)
]

header_parser = HeaderParser(header_rule)

statement_parser = preset.default_statement_parser()

parser = LogParser(header_parser, statement_parser, ignore_failure=True)

