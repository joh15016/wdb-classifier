# ZTF photometry and WDB features

## Retrieve numeric ZTF sources

Supply a CSV with one `ztf_id` or `source_id` column containing numeric ZTF
source IDs. Other columns, including labels, are ignored. Alert names such as
`ZTF19...` are not interchangeable with numeric database IDs. A numeric source
ID normally represents one observing band, not necessarily one physical system.

```bash
wdb-ztf-fetch --ids data/source_ids.csv --config configs/ztf.toml \
  --output data/ztf_photometry --max-new-sources 1 \
  --credentials-file /path/to/private/credentials.yaml
```

The external YAML file contains `hosts.gloria.username` and
`hosts.gloria.password` for a configured `gloria.caltech.edu` server. Only the
matching host entry is read. No credential file is discovered automatically.
Use a private file outside the repository with restrictive permissions.

Alternatively, export `WDB_ZTF_USERNAME`/`WDB_ZTF_PASSWORD` and select
`--password-login`, or export `WDB_ZTF_TOKEN` and omit the password options.
`--token-env NAME` selects another token variable. Do not put secret values in
commands, configs, data tables or source control. Password-login tokens remain
in process memory. A failed token does not silently switch accounts.

Use an explicit host and collection in `configs/ztf.toml`; there is no automatic
release selection. Confirm access before running a large retrieval. The fetcher
keeps `catflags == 0`, finite measurements, positive errors, configured bands
and program IDs, an optional HJD cutoff, and the configured minimum spacing
between retained measurements. The default spacing is five minutes; consider
its effect on short eclipses before adopting it for a scientific search.
An epoch without program metadata is not rejected solely for that absence.

Outputs are `photometry.parquet`, `retrieval_audit.parquet`, a status record,
and per-source checkpoints. Successful, empty and filtered results are distinct;
failed queries are never cached as successful empty matches. Repeat the command
to process more IDs. Completed sources reload without credentials or a network
request. Use a new output directory to intentionally refresh cached observations.

The token-only interface also remains available as
`python -m wdb_classifier.photometry fetch`. Authentication and cleaning do not
require Hyrax or changes to any dependency. No positional merging or astrophysical
candidate cuts are performed during retrieval.

## Photometry contract

Tables require `source_id, band, time, mag, mag_err`. Epochs are float64 days
in one declared time system; no automatic time/flux conversion is performed.
Retain available `ra`, `dec` and time-system metadata. If one ID has several
bands, period extraction chooses the band with the most valid epochs, with
input order breaking ties. Different source IDs are not merged automatically.

Use `scripts/import_photometry.py --help` to normalize local table column names
without querying a survey. This does not transform time systems or measurements.

## Period, Fourier and eclipse features

```bash
python -m wdb_classifier.photometry extract \
  --photometry data/ztf_photometry/photometry.parquet \
  --config configs/period_eclipse.toml --output runs/period_features \
  --max-new-sources 1
```

The default profile searches LS, conditional entropy and AoV over five minutes
to ten days, retaining ten peak hypotheses per algorithm. Consensus and alias
handling use multiple peaks, harmonics and sidereal families. Inspect these
rules when defining the target period range; a selected photometric period is
not guaranteed to equal the orbital period.

The output schema `wdb_period_eclipse50_v1` contains 16 period/Fourier fields
and 34 eclipse fields. Eclipse measurements use Astropy Box Least Squares at
selected and alternate peak/harmonic hypotheses, with minimum point, night,
cycle and leave-one-out support. Labels, coordinates and known catalog periods
are not accepted by these feature calculations.

Outputs include `features.parquet`, `peaks.parquet`, `schema.json`, status and
per-source checkpoints. Match feature definitions and source/band selection
between training and inference. These are not automatically joined with catalog
or magnitude-statistics tables, and no fitted classifier is supplied by extraction.

The grid uses the whole supplied input's baseline, not only the resume chunk.
Use the same explicit `--baseline-days` across independent shards and separate
output directories. `--max-trials` prevents unexpectedly large grids. The CLI
uses one source at a time, one CPU thread by default (up to four explicitly),
and a 4 GiB address-space cap. The time budget is checked between sources.

## Reuse your own period results

Add `--cached-peaks path/to/peaks.parquet` to recompute Fourier/consensus and
eclipse features without searching the period grid. The cache must have unique
source/band keys, all requested peak fields, and matching fingerprints of the
ordered input photometry. Source IDs alone do not establish cache compatibility.

## Reuse precomputed ZVAR FPW results

ZVAR can supply external FPW period hypotheses when generating features. It
does not replace the light curve: Fourier and eclipse measurements still use
the supplied photometry. Install the `zvar` extra and export your existing
`ZVAR_USERNAME` and `ZVAR_PASSWORD` locally. No credentials belong in the config.

First cache candidate periods using photometry's `source_id, ra, dec`:

```bash
python -m wdb_classifier.photometry zvar-fetch \
  --photometry data/ztf_photometry/photometry.parquet \
  --config configs/zvar.toml --output data/zvar_periods --max-new-sources 20
```

Repeat to finish the desired cohort. Cached successful lookups, including empty
and ambiguous matches, are reused without credentials or network access. Failed
HTTP/authentication or malformed responses stop the fetch and are **not** cached
as missing sources. The fetcher requests no photometry and sends no labels,
classifier scores or known orbital periods. Missing coordinates are recorded
explicitly. Multiple PS1 counterparts are marked ambiguous rather than silently
choosing the closest. Default search/acceptance radii are 3/1.5 arcsec.

Use the cached FPW results during extraction:

```bash
python -m wdb_classifier.photometry extract \
  --photometry data/ztf_photometry/photometry.parquet \
  --config configs/period_eclipse.toml --output runs/features_with_zvar \
  --zvar-cache data/zvar_periods --zvar-mode supplement --max-new-sources 20
```

There are two explicit policies:

- `supplement` (default when a ZVAR cache is supplied): keep the configured local
  searches and their selected-period Fourier fit; add ZVAR FPW metadata and its
  period/harmonic hypotheses to the eclipse measurements. Local algorithms must
  exclude FPW in this mode so FPW is not redundantly searched. If no usable ZVAR
  result exists, ordinary local features are still computed.
- `prefer`: use available ZVAR candidates for the selected-period Fourier fit
  and eclipse hypotheses **without allocating or searching a dense local grid**.
  Sources lacking an unambiguous usable result fall back to the configured local
  searches. A matching `--cached-peaks` table can supply those fallback results
  without regenerating expensive periodograms.

Both policies create a new, explicit 57-field schema: the original 50 fields
plus `zvar_fpw_available`, `zvar_fpw_period` (days), `zvar_fpw_significance`
(raw FPW score), `zvar_fpw_fap`, `zvar_fpw_n_sigma`, `zvar_fpw_bin`, and
`zvar_fpw_candidate_count`. Missing FPW measurements remain missing, with an
availability indicator. In `prefer` mode, `period_significance` is missing for
ZVAR-selected periods; an FPW score is never substituted into the local-search
significance field. **Train/evaluate a model with the chosen profile; do not
silently swap it into a model trained on the original features.**

Candidates are restricted to the configured period range, and flagged duplicate
frequencies are excluded. There is no minimum-significance cut or source rejection.
The lowest calibrated FAP selects the representative FPW period; if FAP is absent,
the deterministic fallback prefers the finest bin, then its raw significance.
All retained candidates, not just that representative, feed eclipse measurements.
`--zvar-bands any_gr` (default) allows both g and r hypotheses, including when our
most-observed band is i; `same_band` restricts candidates to the selected band.
These are external hypotheses from different data, not guaranteed phase-coherent
periods for the input light curve. No HJD/BJD conversion or period refinement is
performed. The API's returned candidates need not be the complete on-disk peak set.

`features.parquet` records `period_provider` and `zvar_status` as provenance, not
classifier fields. `zvar_candidates.parquet` preserves PS1 ID, band, field,
CCD/quadrant, bin and FPW statistics. `peaks.parquet` contains only actual local
search/replay rows, never fabricated local peaks for ZVAR-backed sources.

Finish or freeze the desired lookup cache before extraction: source-cache hashes,
matching policy, code, feature mode and photometry fingerprints are checked on
resume. Changes require a new feature output directory. Existing periodograms
can still be reused with `--cached-peaks`; do not erase prior run directories.
Extraction itself is offline once lookup results are cached. Neither a missing
ZVAR counterpart nor a weak FPW period means that a source is physically not a WDB.

See the [ZVAR DR1 paper](https://arxiv.org/html/2611.16166v1) and the
[official API documentation](https://api.zvar.caltech.edu/imdiff/docs) for the
external data products. This integration supplies features; it does not establish
that classifier recall or false-positive rates are unchanged.

## Optional magnitude statistics

```bash
wdb-classifier features --photometry data/ztf_photometry/photometry.parquet \
  --output data/magnitude_statistics.parquet --time-system HJD_as_returned
```

The independent `light-curve` provider computes 16 namespaced statistics per
band plus quality diagnostics over the entire supplied baseline. It removes
invalid/nonpositive-error measurements, sorts epochs, keeps the first valid
duplicate epoch, and performs no sigma clipping or transient-window truncation.
The table and manifest state units, definitions and software version. Source-level
aggregation and the classifier's chosen schema are explicit scientific choices.
