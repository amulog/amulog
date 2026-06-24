# amulog loghub examples

Ready-to-run amulog setups for the [loghub](https://github.com/logpai/loghub)
open log datasets. Each `loghub_<Name>/` directory builds an amulog SQLite DB
from a 2,000-line sample, two ways:

- **`config.conf`** — `lt_methods = drain`: generate log templates from the logs.
- **`config_import.conf`** — `lt_methods = import`: classify the logs with the
  bundled amulog templates `<Name>.tpl` (derived from the loghub ground truth).

This mirrors the per-dataset parser examples in
[log2seq](https://github.com/amulog/log2seq) `example/loghub_*`: amulog reuses
those `parser.py` scripts (via `manager.parser_script`) to extract the host and
timestamp. See `NOTICE` for data/template attribution (loghub is CC BY 4.0).

## Directory layout

```
loghub_<Name>/
  parser.py            # log2seq parser (copied; defines `parser`)
  <Name>_2k.log        # loghub 2k sample (CRLF / trailing space normalized)
  <Name>.tpl           # amulog template definitions (lt-import format)
  config.conf          # lt_methods = drain   -> generate templates
  config_import.conf   # lt_methods = import  -> classify with <Name>.tpl
```

The DB (`log.db`) and other generated artifacts are git-ignored — running a
config plus the bundled `<Name>_2k.log` reproduces the same DB on any machine.

## Run

Generate templates from the logs (drain):

```sh
cd loghub_BGL
python -m amulog db-make -c config.conf
python -m amulog show-db-info -c config.conf   # lines / hosts / templates
python -m amulog show-lt -c config.conf        # the generated templates
```

Or classify the same logs with the bundled amulog templates (import):

```sh
python -m amulog db-make -c config_import.conf
python -m amulog show-db-info -c config_import.conf
```

`<Name>.tpl` is amulog's own template format (one segmented template per line, as
produced by `show-lt-import`). It is derived from the loghub ground truth, not a
copy of it — see "How the templates are produced" below.

## Status

`drain` = templates auto-generated; `import` = templates in `<Name>.tpl` (every
2k line is classified with no `lt_fail` in both modes). `hosts=1` means the
parser extracts no host (amulog uses `dummy`).

| Dataset | hosts | drain tpl | import (`.tpl`) |
|---|--:|--:|--:|
| Android | 1 | 148 | 171 |
| Apache | 1 | 6 | 6 |
| BGL | 1778 | 99 | 115 |
| Hadoop | 1 | 124 | 114 |
| HDFS | 1 | 146 | 16 |
| HPC | 1 | 46 | 51 |
| Linux | 1 | 112 | 117 |
| Mac | 38 | 382 | 344 |
| OpenSSH | 1 | 23 | 27 |
| OpenStack | 10 | 115 | 46 |
| Proxifier | 1 | 538 | 12 |
| Spark | 1 | 62 | 36 |
| Thunderbird | 491 | 201 | 153 |
| Windows | 1 | 57 | 50 |
| Zookeeper | 1 | 47 | 53 |

Excluded (no directory): **HealthApp** (its loghub parser leaves the timestamp
unparsed — the abbreviated date is ambiguous — and amulog requires a timestamp).

## Use the full datasets

The examples ship only the 2k samples. To run on a full dataset, obtain it from
loghub, point a config's `src_path` at it (a file or a directory with
`src_recur = true`), and `db-make` again. Full logs and DBs are large and are
never committed.

## How the templates are produced

`<Name>.tpl` is generated, not hand-written: the loghub ground-truth templates
(`<Name>_2k.log_templates.csv`, which use the placeholders `<*>` and, for Mac,
`<NUM>`) are loaded with `lt_methods = import-ext` and
`log_template_import.ext_replacer = <*>, <NUM>`, which matches each placeholder
against the raw message and re-segments it into amulog tokens; the result is
dumped with `show-lt-import`. This conversion, the ground-truth CSVs, and the
procedure are maintained outside this public example (they require the logparser
benchmark repo). The committed `<Name>.tpl` is sufficient to run the import
example as-is.
