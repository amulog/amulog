# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
