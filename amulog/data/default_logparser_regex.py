#!/usr/bin/env python
# coding: utf-8

"""Amulog.logparser loads following objects:
* header_regex_list (list of re.SRE_Pattern)
* split_regex_first (re.SRE_Pattern)
* fix_ipaddr (boolean)
* fix_ipnet (boolean)
* fix_regex_list (list of re.SRE_Pattern)
* split_regex_second (re.SRE_Pattern)

* header_regex_list (list of re.SRE_Pattern)
Most top matched object is used for parsing log message header
(timestamp and hostname).
Each re.SRE_Pattern should provide named groups in following keys.
- year (digit)
- month (digit) or bmonth (abbreviated month name like "Jan")
- day (digit)
- hour (digit)
- minute (digit)
- second (digit)
- host (str)
- message (str)

* split_regex_first (re.SRE_Pattern)
Matched string is recognized as a symbol string.
Give capturing group like r"([ ,\.])".
split_regex_second is a special case,
so use split_regex_first in usual case.

If a symbol is not always a splitter, use fix_foo and split_regex_second.
e.g., ":" is usually a splitter, but ":" in IPv6 addr is not a splitter.
In this case, use fix_ipaddr = True and specify ":" as split_regex_second.
Fixed words are not splitted by symbols in split_regex_second.
"""

import re

_restr_timestamp = (r"(?P<year>\d{4}\s+)?"        # %Y_ (optional)
                   r"(?P<bmonth>[a-zA-Z]{3})\s+" # %b_
                   r"(?P<day>\d{1,2}\s+)"        # %d_
                   r"(?P<hour>\d{2}):"           # %H:
                   r"(?P<minute>\d{2}):"         # %M:
                   r"(?P<second>\d{2})")         # %S
_restr_datetime = (r"(?P<year>\d{4})-"   # %Y-
                  r"(?P<month>\d{2})-"  # %m-
                  r"(?P<day>\d{2})\s+"  # %d_
                  r"(?P<hour>\d{2}):"   # %H:
                  r"(?P<minute>\d{2}):" # %M:
                  r"(?P<second>\d{2})") # %S
_restr_host = (r"([a-zA-Z0-9][a-zA-Z0-9:.-]*[a-zA-Z0-9]" # len >= 2
              r"|[a-zA-Z0-9])")                         # len == 1
_restr_syslog_ts = (r"^{0}\s+(?P<host>{1})\s*(?P<message>.*)$".format(
    _restr_timestamp, _restr_host))
_restr_syslog_dt = (r"^{0}\s+(?P<host>{1})\s*(?P<message>.*)$".format(
    _restr_datetime, _restr_host))

_split_regex_first = re.compile(r"([\(\)\[\]\{\}\"\|\+',=><;`# ]+)")
_split_regex_second = re.compile(r"([:]+)")
_restr_time = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$")
_restr_mac = re.compile(r"^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$")

header_list = [re.compile(_restr_syslog_ts),
               re.compile(_restr_syslog_dt)]
split_list = [('split_regex', _split_regex_first),
              ('fix_ip', (True, True)),
              ('fix_regex', [_restr_time, _restr_mac]),
              ('split_regex', _split_regex_second)]

