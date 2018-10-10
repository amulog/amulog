#!/usr/bin/env python
# coding: utf-8

import pyparsing as pp
#import cPyparsing as pp

default_year = 2018
split_sym = " ()[]',+=\"><{};`|#"

d2int = pp.Word(pp.nums, exact = 2)
c3month = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
cmonth = pp.Or([pp.Literal(c) for c in c3month])
year = pp.Word(pp.nums, exact = 4)
month = pp.Word(pp.nums, exact = 2)

day = pp.Word(pp.nums, max = 2)
#nday = pp.NotAny("0") + pp.Word(pp.nums, max = 2)
#day = d2day ^ nday

dash = pp.Suppress("-")
colon = pp.Suppress(":")
dates = year + dash + month + dash + day
times = d2int + colon + d2int + colon + d2int

timestamp_syslog = pp.Optional(year, default = str(default_year)) + \
                   cmonth + day + times
timestamp_datetime = dates + times
timestamp = timestamp_syslog ^ timestamp_datetime
#timestamp = timestamp_syslog
timestamp.setName("timestamp")

hostname = pp.Word(pp.alphanums + "-" + "." + ":")
hostname.setName("hostname")

message_word = pp.CharsNotIn(split_sym)
message_sym = pp.Optional(pp.Word(split_sym), default = "")
message = message_sym + pp.ZeroOrMore(message_word + message_sym)
hostname.setName("message")

header = timestamp + hostname
logmessage = header + message
#logmessage.enablePackrat()

if __name__ == "__main__":
    import sys
    import datetime
    if len(sys.argv) < 2:
        sys.exit("usage: {0} filename".format(sys.argv[0]))
    fn = sys.argv[1]

    with open(fn, 'r') as f:
        for line in f:
            line = line.rstrip("\n")
            ret = logmessage.parseString(line)

            v_dt = ret[0:6]
            if v_dt[1] in c3month:
                v_dt[1] = c3month.index(v_dt[1]) + 1
            dt = datetime.datetime(*[int(i) for i in v_dt])
            host = ret[6]
            l_s = ret[7::2]
            l_w = ret[8::2]
            print("{0} {1} {2} {3}".format(dt, host, l_w, l_s))


#s = "2018-09-03 01:24:00"
#s = "Jun  9 01:24:00 hoge.hongo.wide.ad.jp"
#    ret = header.parseString(s)
#    import datetime
#    if ret[1] in c3month:
#        ret[1] = c3month.index(ret[1]) + 1
#    dt = datetime.datetime(*[int(i) for i in ret[0:6]])
#    print(dt)
#    print(ret[6])






#import time
#print(time.strftime("%Y-%m-%d %H:%M:%S"))

#word = pp.Word(pp.alphanums)
#sh = pp.Literal("#")
#supp = pp.Suppress(sh)
#white = pp.Suppress(pp.White(" "))
#parser = word + white + supp + word
#parser2 = parser + supp + word
#parser2.leaveWhitespace()
#
#s = "hoge   #john#hage"
#print(parser2.parseString(s))

