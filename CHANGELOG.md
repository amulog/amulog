# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `log_template_import.ext_replacer`: configure the variable placeholder(s)
  used in an import-ext template-definition file (comma-separated, e.g.
  `<*>, <NUM>` for loghub). The placeholders are matched verbatim and each
  captures a non-greedy span, so a literal `*` in a template or matched value
  (`****`, `tag="*launch*"`, `******* GOODBYE`) is no longer confused with
  amulog's `**` wildcard. Empty (default) keeps the existing `**`/`*NAME*`
  behavior.
- `example/loghub_*`: ready-to-run amulog setups for the loghub open log
  datasets (2k samples). Each directory has the log2seq parser, the 2k sample,
  amulog-form template definitions (`<Name>.tpl`, derived from the loghub ground
  truth), and two configs: `config.conf` (generate templates with drain) and
  `config_import.conf` (classify the logs with `<Name>.tpl`).
- `general.timezone`: select the timezone amulog uses to interpret stored log
  timestamps. Timestamps are stored as the device's naive wall-clock time; this
  option only labels the datetimes amulog returns and how datetime query bounds
  are interpreted (stored values are unchanged). Empty (default) or `local` =
  system local timezone (current behavior); `utc` or an IANA zone name are also
  accepted. Replaces the previously hardcoded `tzlocal()` in `_str2datetime`,
  `config.getdt` and `config.getterm`.
- `show-algorithms`: list the registered log template generation algorithms
  with their classification (stateful, online/offline, parallel safety)
  derived from code — the online/offline mode from the `alg/meta.py`
  registration lists and stateful/parallel from the `lt_common.py` class
  hierarchy. Use `--format json` for machine-readable output. This replaces
  hand-maintained algorithm tables that tend to drift.

### Changed
- **Breaking:** amulog now escapes its template-special characters (`\`, `*`,
  `@`) on the parsed statement (words and message) *after* the log2seq header is
  parsed, instead of escaping the whole raw line *before* parsing. Header fields
  (host, timestamp) keep their raw value. Previously a header containing `@`/`*`
  (e.g. a `user@host` location) was escaped, which made log2seq fail to parse
  the line and amulog silently dropped it (e.g. ~1384/2000 loghub Thunderbird
  lines); such lines are now parsed and stored, and host values with those
  characters are stored raw rather than backslash-escaped. Statement `words` are
  byte-identical for lines that previously parsed, so template/eval outputs do
  not change for clean-header data. Because stored data can differ, this warrants
  a minor version bump (→ 0.4.0); keep old log2seq-parser/config assets on the
  old amulog version.
- Tests are split into the core suite under `tests/` (run by CI) and external
  tests under `tests-ext/` that require optional backends and live servers (e.g.
  the MySQL backend needs `pymysql` and a running MySQL). `pytest.ini` sets
  `testpaths = tests`, so a bare `pytest` runs only the core suite; the external
  tests are opt-in via `pytest tests-ext/`.

### Fixed
- `manager.normalize_pline`: a log2seq parser with an optional host field can
  emit `host = None`; this reached `host_alias.resolve_host(None)` and crashed
  with `AttributeError`. A `None` host is now treated like a missing host and
  falls back to `dummy_host`.
- `manager`: the optional `lid` field is now read with `pline.get("lid") is not
  None` instead of `"lid" in pline`, so a parser emitting `lid = None` (present
  but null) no longer raises `int(None)`.

## [0.3.14] - 2026-06-16

This is a maintenance release: mostly bug fixes, with a few minor additions.

### Added
- `show-db-info -t/--term START END`: show database status (counts of lines,
  templates, template groups and hosts) restricted to a datetime range.
- Optional-dependency extras: `pip install amulog[ssdeep]` and
  `pip install amulog[mysql]` install the optional ssdeep grouping / MySQL backends.

### Changed
- `amsemantics` is now imported lazily and treated as optional; when it is missing,
  a clear actionable error is raised instead of a bare `ImportError`.
- Datetime query keys: `dts`/`dte` are the canonical keyword arguments for time
  ranges; the legacy `top_dt`/`end_dt` are still accepted as aliases, with the
  compatibility conversion centralized in one place.
- CI migrated from Travis to GitHub Actions (test matrix on Python 3.8-3.12, PyPI
  trusted publishing).
- Template-definition files are documented as a trust boundary (their regexes are
  not sandboxed and have no match timeout); project metadata updated (repository URL
  moved to the `amulog` org, author email).

### Fixed
- `ltgroup_alg = ssdeep` could not be instantiated at all (`LTGroupFuzzyHash`); it
  now works as an online grouping method.
- Crashing CLI / algorithm paths:
  - `edit show-lt-breakdown` passed a config object instead of `LogData`.
  - Unknown subcommands now print usage instead of raising `KeyError`.
  - `fttree` offline processing returned `(tid, state)` tuples, breaking the
    offline database build.
  - `shiso` grouping could not be instantiated, ignored an explicit `cfunc`, and
    raised `IndexError` on gapped template ids.
  - `crf` normalizer path tuple-indexed a list (`labels[start, stop]`) and left a
    `pdb.set_trace()` on a production path.
  - Parallel offline processing dereferenced `None` (`self._ltgen`) in
    `process_offline` / `load` / `clean`.
- Log-template search tree (`lt_search`) `remove`: stack-backtrack pointer, prune
  key, `None`-guard and wildcard-node bugs.
- Evaluation metrics: zero-division producing `nan` in precision/recall/F-score;
  label-dependent `over_aggregation_cluster_ratio`; misaligned answer/trial labels
  when their `None` positions differed.
- Config: circular `import` directives now raise instead of looping forever;
  `release_common_logging` ignored a list argument; `str2dur` malformed-input
  handling.
- `data-from-data` used a wrong config section and skipped message escaping.
- Parallel offline batching used the configured batch *size* as the batch *count*.
- `fail_dump` now writes UTF-8 and ensures one trailing newline per record.
- Numerous silent bugs across lenma, VA, crf, config, host_alias, lt_common,
  `Timer.stat` and `template_from_messages`.

### Removed
- Dead `LTPostProcess.search` (called a method that no longer exists) and the unused
  `recalculation=True` code path in evaluation.
