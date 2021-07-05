
"""argparse wrapper"""

import os
import sys
import argparse


def main(d_argset, d_alias=None, sublibs=None):
    command_path = os.path.basename(sys.argv[0])

    buf_usage = [
        ("usage: " + command_path + " SUBCOMMAND [options and arguments] ..."),
        "",
        "subcommands: "
    ]

    d_subc = {}
    for key, argset in d_argset.items():
        d_subc[key] = argset[0].partition("@")[0]
    if d_alias:
        for alias, key in d_alias.items():
            d_subc[alias] = "same as {0}".format(key)

    buf_usage += ["  {0}: {1}".format(k, v)
                  for k, v in sorted(d_subc.items())]

    buf_usage += [
        "",
        ("try \"" + command_path + " SUBCOMMAND -h\" "
         "to refer detailed subcommand usage")
    ]

    if sublibs:
        buf_usage.append(
            "see also sub-libraries {0}".format(
                " ".join(["amulog.{0}".format(n) for n in sublibs])
            )
        )

    usage = "\n".join(buf_usage)

    if d_alias:
        for alias, key in d_alias.items():
            d_argset[alias] = d_argset[key]

    if len(sys.argv) < 2:
        sys.exit(usage)
    mode = sys.argv[1]
    if mode in ("-h", "--help"):
        sys.exit(usage)
    commandline = sys.argv[2:]

    desc, l_argset, func = d_argset[mode]
    ap = argparse.ArgumentParser(prog=" ".join(sys.argv[0:2]),
                                 description=desc.replace("@", ""))
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args(commandline)
    func(ns)
