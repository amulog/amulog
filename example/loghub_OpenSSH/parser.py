#!/usr/bin/env python
# coding: utf-8

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rules = [
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    Hostname("host"),
    UserItem("component", r"[a-zA-Z0-9()._-]+"),
    Digit("processid", optional=True),
    Statement()
]

defaults = {"year": datetime.datetime.now().year}

header_parser = HeaderParser(header_rules, separator=" :[]", defaults=defaults)

statement_parser = preset.default_statement_parser()

parser = LogParser(header_parser, statement_parser)

