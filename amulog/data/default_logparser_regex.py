#!/usr/bin/env python
# coding: utf-8

"""Amulog.logparser loads regex_list (list of re.SRE_Pattern]) here.
each re.SRE_Pattern should provide named groups in following keys.
- year (digit)
- month (digit) or bmonth (abbreviated month name)
- day (digit)
- hour (digit)
- minute (digit)
- second (digit)
- host (str)
- message (str)
"""

restr_timestamp = (r"(?P<year>\d{4}\s+)?"        # %Y_ (optional)
                   r"(?P<bmonth>[a-zA-Z]{3})\s+" # %b_
                   r"(?P<day>\d{1,2}\s+)"        # %d_
                   r"(?P<hour>\d{2}):"           # %H:
                   r"(?P<minute>\d{2}):"         # %M:
                   r"(?P<second>\d{2})")         # %S
restr_datetime = (r"(?P<year>\d{4})-"   # %Y-
                  r"(?P<month>\d{2})-"  # %m-
                  r"(?P<day>\d{2})\s+"  # %d_
                  r"(?P<hour>\d{2}):"   # %H:
                  r"(?P<minute>\d{2}):" # %M:
                  r"(?P<second>\d{2})") # %S
restr_host = (r"([a-zA-Z0-9][a-zA-Z0-9:.-]*[a-zA-Z0-9]" # len >= 2
              r"|[a-zA-Z0-9])")                         # len == 1
restr_syslog_ts = (r"^{0}\s+{1}\s*(?P<message>.*)$".format(restr_timestamp,
                                                           restr_host))
restr_syslog_dt = (r"^{0}\s+{1}\s*(?P<message>.*)$".format(restr_datetime,
                                                           restr_host))

regex_list = [re.compile(restr_syslog_ts),
              re.compile(restr_syslog_dt)]

