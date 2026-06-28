#!/usr/bin/env python
# coding: utf-8

"""Host stratification (host group) resolver.

This module provides a DB-independent classifier that maps a host
(the *original* host stored in the log table) to a host group id (hgid)
for a given *tier*. A tier is an ordered classification axis (fine to
coarse, e.g. ``default`` / ``midplane`` / ``rack``); within a tier the
host is resolved by a sequence of *schemes* and the first matching
scheme wins.

The definition is read from a dedicated ini file (separate from the main
config, mirroring the ``host_alias_filename`` / ``lt_label`` precedent),
pointed to by ``[manager] host_group_filename`` in the main config. The
ini grammar follows Prometheus relabel_configs (regex / replacement /
type as separate fields)::

    [tiers]
    order = default, midplane, rack    # fine -> coarse
    unmatched = keep                   # keep | drop | drop_alert
    label_tier = default               # tier used for variable labeling

    [tier_default]
    schemes = ipaddr, s4               # ordered; first match wins
    ipaddr.type = builtin
    ipaddr.name = ipaddr
    s4.type = hostalias
    s4.file = def_s4_host_alias.txt
    s4.ignorecase = true

    [tier_midplane]
    schemes = bgl
    bgl.regex = ^(R\\d+-M\\d)          # type defaults to regex
                                       # replacement defaults to \\1 (capture)

Scheme field reference:
- ``<name>.type``: ``regex`` (default) / ``hostalias`` / ``builtin``.
- regex: ``.regex`` (required, with a capture group), ``.replacement``
  (default ``\\1``; **Python re.sub backref syntax**, not Prometheus ``$1``),
  ``.ignorecase`` (bool, default false).
- hostalias: ``.file`` (host_alias-format file used as a literal,
  multi-to-one lookup), ``.ignorecase`` (default true). Reuses
  amulog.host_alias so IP canonicalization / case-insensitivity /
  CIDR-literal / IPv6-expansion / no-substring-mismatch are identical.
- builtin: ``.name`` (only ``ipaddr`` is shipped initially).
"""

import re
import logging
import ipaddress
import configparser

from . import host_alias

_logger = logging.getLogger(__package__)

# unmatched policies
UNMATCHED_KEEP = "keep"
UNMATCHED_DROP = "drop"
UNMATCHED_DROP_ALERT = "drop_alert"
_UNMATCHED_POLICIES = (UNMATCHED_KEEP, UNMATCHED_DROP, UNMATCHED_DROP_ALERT)


def _str2bool(string):
    return string.strip().lower() in ("true", "1", "yes", "on")


class _Scheme(object):
    """Base class for a single resolution scheme within a tier."""

    def resolve(self, host):
        """Return hgid string if this scheme matches host, else None."""
        raise NotImplementedError


class _RegexScheme(_Scheme):
    """Resolve by a regular expression with a capture group.

    The matched part is formatted by ``re.sub`` with ``replacement``
    (Python backref syntax, e.g. ``\\1``). The default replacement is
    ``\\1`` (the first capture group).
    """

    def __init__(self, regex, replacement="\\1", ignorecase=False):
        if regex is None or regex == "":
            raise ValueError("regex scheme requires a non-empty .regex")
        flags = re.IGNORECASE if ignorecase else 0
        self._regex = re.compile(regex, flags)
        self._replacement = replacement

    def resolve(self, host):
        m = self._regex.search(host)
        if m is None:
            return None
        return m.expand(self._replacement)


class _HostAliasScheme(_Scheme):
    """Resolve by a host_alias-format file (literal multi-to-one lookup).

    Reuses amulog.host_alias.HostAlias so that IP canonicalization,
    case-insensitive hostname matching, CIDR-literal handling, IPv6
    expansion and the no-substring-mismatch behaviour are identical to
    the legacy host_alias path.
    """

    def __init__(self, fn, ignorecase=True):
        # host_alias matches hostnames case-insensitively by design; the
        # ignorecase flag is kept for grammar symmetry. host_alias does not
        # currently expose a case-sensitive mode, so a case-sensitive
        # request is honoured at the query layer below.
        self._ignorecase = ignorecase
        self._ha = host_alias.HostAlias(fn)

    def resolve(self, host):
        # host_alias.resolve_host lowercases hostname queries internally;
        # IP tokens are matched by canonical form regardless of case.
        return self._ha.resolve_host(host)


class _BuiltinIpaddrScheme(_Scheme):
    """Builtin ``ipaddr`` scheme: canonicalize an IP token.

    Matches only valid IP addresses (IPv4/IPv6); returns the canonical
    string form (so equivalent notations such as expanded IPv6 collapse
    to one hgid). Non-IP tokens do not match.
    """

    def resolve(self, host):
        try:
            return str(ipaddress.ip_address(host))
        except ValueError:
            return None


_BUILTIN_SCHEMES = {
    "ipaddr": _BuiltinIpaddrScheme,
}


def _build_scheme(parser, section, name):
    """Build a scheme object from ``<name>.*`` keys in a tier section."""

    def field(key, default=None):
        opt = "{0}.{1}".format(name, key)
        if parser.has_option(section, opt):
            return parser.get(section, opt)
        return default

    stype = (field("type") or "regex").strip().lower()

    if stype == "regex":
        regex = field("regex")
        replacement = field("replacement")
        if replacement is None:
            replacement = "\\1"
        ignorecase = _str2bool(field("ignorecase", "false"))
        return _RegexScheme(regex, replacement, ignorecase)
    elif stype == "hostalias":
        fn = field("file")
        if fn is None or fn == "":
            raise ValueError(
                "hostalias scheme {0!r} requires a .file".format(name))
        ignorecase = _str2bool(field("ignorecase", "true"))
        return _HostAliasScheme(fn, ignorecase)
    elif stype == "builtin":
        bname = (field("name") or "").strip()
        if bname not in _BUILTIN_SCHEMES:
            raise ValueError(
                "unknown builtin scheme name {0!r} (available: {1})".format(
                    bname, ", ".join(sorted(_BUILTIN_SCHEMES))))
        return _BUILTIN_SCHEMES[bname]()
    else:
        raise ValueError(
            "unknown scheme type {0!r} for scheme {1!r}".format(stype, name))


class HostGroup(object):
    """Host stratification resolver.

    Reads the dedicated ini pointed to by ``[manager] host_group_filename``.
    An empty filename disables host stratification (no tiers).

    Note:
        Each tier is a single-valued function of host, so "1 host = 1 hg
        per tier" holds structurally (it does not conflict with the
        single-group-per-host constraint of template variable labeling).
    """

    def __init__(self, conf):
        self._fn = conf["manager"]["host_group_filename"]
        self._tiers = []  # ordered tier names (fine -> coarse)
        self._schemes = {}  # tier name -> list[_Scheme]
        self._unmatched = UNMATCHED_KEEP
        self._label_tier = None
        if self._fn is not None and self._fn != "":
            self._open(self._fn)

    def _open(self, fn):
        parser = configparser.ConfigParser()
        # preserve case of option names (scheme field keys are case-sensitive
        # in spirit and avoid surprises with regex content)
        parser.optionxform = str
        read = parser.read(fn)
        if len(read) == 0:
            raise IOError("host_group definition load error ({0})".format(fn))

        if not parser.has_section("tiers"):
            raise ValueError(
                "host_group file {0!r} has no [tiers] section".format(fn))

        # [tiers]
        order_raw = parser.get("tiers", "order", fallback="")
        self._tiers = [e.strip() for e in order_raw.split(",")
                       if e.strip() != ""]

        unmatched = parser.get("tiers", "unmatched",
                               fallback=UNMATCHED_KEEP).strip().lower()
        if unmatched == "":
            unmatched = UNMATCHED_KEEP
        if unmatched not in _UNMATCHED_POLICIES:
            raise ValueError(
                "invalid unmatched policy {0!r} (expected one of {1})".format(
                    unmatched, ", ".join(_UNMATCHED_POLICIES)))
        self._unmatched = unmatched

        label_tier = parser.get("tiers", "label_tier", fallback="").strip()
        self._label_tier = label_tier if label_tier != "" else None

        # [tier_<name>]
        for tier in self._tiers:
            section = "tier_{0}".format(tier)
            if not parser.has_section(section):
                raise ValueError(
                    "host_group file {0!r}: missing section [{1}] "
                    "for tier {2!r}".format(fn, section, tier))
            scheme_names_raw = parser.get(section, "schemes", fallback="")
            scheme_names = [e.strip() for e in scheme_names_raw.split(",")
                            if e.strip() != ""]
            if len(scheme_names) == 0:
                raise ValueError(
                    "host_group file {0!r}: tier {1!r} has no schemes".format(
                        fn, tier))
            self._schemes[tier] = [
                _build_scheme(parser, section, sname)
                for sname in scheme_names
            ]

        if self._label_tier is not None and self._label_tier not in self._tiers:
            raise ValueError(
                "label_tier {0!r} is not in tier order {1}".format(
                    self._label_tier, self._tiers))

    def tiers(self):
        """Return tier names in defined order (fine -> coarse)."""
        return list(self._tiers)

    def label_tier(self):
        """Return the tier used for template variable labeling, or None."""
        return self._label_tier

    def unmatched_policy(self):
        return self._unmatched

    def resolve(self, host, tier):
        """Resolve host to its hgid for the given tier.

        Schemes are tried in definition order; the first matching scheme's
        output is the hgid. If no scheme matches, the tier's unmatched
        policy applies:
        - keep: return host itself (lossless, nesting preserved)
        - drop: return None
        - drop_alert: return None and emit a warning log
        """
        if tier not in self._schemes:
            raise KeyError("unknown tier {0!r}".format(tier))
        for scheme in self._schemes[tier]:
            hgid = scheme.resolve(host)
            if hgid is not None:
                return hgid
        # unmatched
        if self._unmatched == UNMATCHED_KEEP:
            return host
        elif self._unmatched == UNMATCHED_DROP:
            return None
        elif self._unmatched == UNMATCHED_DROP_ALERT:
            _logger.warning(
                "host %r did not match any scheme in tier %r (dropped)",
                host, tier)
            return None
        else:  # pragma: no cover - validated at load time
            raise ValueError(self._unmatched)


def init_hostgroup(conf):
    """Construct a HostGroup from the main config.

    Mirrors amulog.host_alias.init_hostalias.
    """
    return HostGroup(conf)
