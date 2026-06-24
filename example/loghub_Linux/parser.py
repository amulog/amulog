#!/usr/bin/env python

import datetime

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


defaults = {"year": datetime.datetime.now().year}

# loghub's <Component> can contain spaces and slashes ("syslogd 1.4.1",
# "-- root", "/sbin/mingetty"), so a separator-based component cannot capture it.
# Pin "<component>(\[<pid>\])?: <content>" with full_format and a non-greedy
# component that stops at the first "[pid]:" / ": " boundary.
header_rule1 = [
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    Hostname("host"),
    UserItem("component", r".+?"),
    Digit("processid", optional=True),
    Statement()
]
full_format1 = r"<0> <1> <2> <3> <4>(\[<5>\])?: <6>"
header_parser1 = HeaderParser(header_rule1, full_format=full_format1, defaults=defaults)

# Tag-less syslog meta-lines have no "<Component>: " part, e.g.
#   "Sep 28 09:08:56 combo last message repeated 2 times"
#   "Sep 28 09:09:19 combo exiting on signal 15"
# Model that class explicitly: keep the timestamp/host envelope and take the
# remainder as the message (rather than dropping the whole line to a catch-all).
header_rule2 = [
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    Hostname("host"),
    Statement()
]
header_parser2 = HeaderParser(header_rule2, separator=" ", defaults=defaults)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)

