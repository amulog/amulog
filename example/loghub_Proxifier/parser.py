#!/usr/bin/env python

import datetime

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


header_rule1 = [
    ItemGroup([Digit("month"),
               Digit("day"),
               Time()], separator=" ."),
    UserItem("env", r".+?"),
    Statement()
]
# loghub's <Program> is everything up to the first " - " (it can contain "-",
# e.g. "git-remote-https.exe", and a " *64" suffix). Pin " - " as the delimiter
# with full_format and match the program non-greedily so it stops at that first
# " - " rather than being split on every "-".
full_format1 = r"\[<0>\] <1> - <2>"

header_rule2 = [
    ItemGroup([Digit("month"),
               Digit("day"),
               Time()], separator=" ."),
    Statement()
]

defaults = {"year": datetime.datetime.now().year}

header_parser1 = HeaderParser(header_rule1, full_format=full_format1, defaults=defaults)
header_parser2 = HeaderParser(header_rule2, separator="[] ", defaults=defaults)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)

