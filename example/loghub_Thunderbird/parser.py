#!/usr/bin/env python

from log2seq import LogParser
from log2seq import preset
from log2seq.header import *
from log2seq.statement import *


# Common envelope shared by both rules. host and location include "#" because
# the full dataset uses placeholder node names such as "#1#" and "#8#/#8#" (the
# majority of lines); without "#" those lines fail to match.
timestamp_prefix = [
    UserItem("label", r"-|[A-Z]+"),
    Digit("unixtime", dummy=True),
    ItemGroup([Digit("year"),
               Digit("month", dummy=True),
               Digit("day", dummy=True)],
               separator="."),
    UserItem("host", r"[a-zA-Z0-9:#-]+"),
    MonthAbbreviation(),
    Digit("day"),
    Time(),
    UserItem("location", r"[a-zA-Z0-9/@#-]+", dummy=True),
]

# Rule 1 — loghub log_format:
#   <Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> \
#   <Component>(\[<PID>\])?: <Content>
# <Component> is a free-form field up to the ": " delimiter (it may contain
# spaces, e.g. "Kernel command line", "IA32 emulation"), so match it non-greedily
# and let full_format's ": " fix the boundary. The process id is optional
# (e.g. "dhcpd:" / "logger:" have none).
header_rule1 = timestamp_prefix + [
    UserItem("component", r".+?"),
    Digit("processid", optional=True),
    Statement()
]
full_format1 = r"<0> <1> <2> <3> <4> <5> <6> <7> <8>(\[<9>\])?: <10>"
header_parser1 = HeaderParser(header_rule1, full_format=full_format1)

# Rule 2 — tag-less syslog meta-lines that have no "<Component>: " part, e.g.
#   "... #1#/#1# exiting on signal 15"  (syslogd shutdown notice)
#   "... local@tbird-admin1 mysql_install_db:"  (empty content)
# Keep the same timestamp/host envelope and take the remainder as the message.
header_rule2 = timestamp_prefix + [Statement()]
header_parser2 = HeaderParser(header_rule2)

statement_parser = preset.default_statement_parser()

parser = LogParser([header_parser1, header_parser2], statement_parser)

