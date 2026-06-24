#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


# loghub log_format:
#   <Date> <Time>, <Level>  <Component>  <Content>
header_rule = [
    Date(),
    Time(),
    String("level"),
    String("component"),
    Statement()
]

header_parser = HeaderParser(header_rule, separator=" ,\t")

# CBS logs interleave header-less continuation/sub-record lines that carry no
# "<Date> <Time>," prefix, e.g. "CSIPERF:TXCOMMIT;200" or "  Scavenge (8): ...".
# Take such a line as-is for the message (there is no timestamp to extract).
header_parser_cont = HeaderParser([Statement()], reformat_timestamp=False)

pattern_windows_fullpath = r"[A-Z]:(\\[a-zA-Z0-9.*?_-])+"

statement_rules = [
    Split('"' + "()[]{}|+',=><;`# "),
    FixIP(),
    Fix([preset.pattern_time,
         preset.pattern_macaddr,
         pattern_windows_fullpath]),
    Split(":")
]

statement_parser = StatementParser(statement_rules)

parser = LogParser([header_parser, header_parser_cont], statement_parser)

