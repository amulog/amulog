#!/usr/bin/env python
# coding: utf-8

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


# loghub log_format:
#   <Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>
# Pin this with full_format so "[" / "]" / "(" / ")" are fixed delimiters rather
# than generic separators. With the old separator-based rule, "[" / "]" were
# separators, which (a) dropped a leading "[" of the content and (b) left a
# sub-process "([pid]):" tail inside the message. <Component> and <Address> are
# free-form fields (component may contain spaces, e.g. "BezelServices 255.10"),
# matched non-greedily up to the bracketed pid and the ": " boundary.
header_rule1 = [
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    Hostname("host"),
    UserItem("component", r".+?"),
    Digit("processid"),
    UserItem("address", r"[^)]*", optional=True, dummy=True),
    Statement()
]
full_format1 = r"<0> <1> <2> <3> <4>\[<5>\]( \(<6>\))?: <7>"

header_rule2 = [
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    UserItem("dummy", r"---"),
    Statement()
]

header_rule3 = [
    Statement()
]

defaults = {"year": datetime.datetime.now().year,
            "host": None}

header_parser1 = HeaderParser(header_rule1, full_format=full_format1, defaults=defaults)
header_parser2 = HeaderParser(header_rule2, separator=" :[]", defaults=defaults)
header_parser3 = HeaderParser(header_rule3, separator=" \t", reformat_timestamp=False)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2, header_parser3], statement_parser)

