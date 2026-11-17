# Architecture

WDB Classifier separates reusable infrastructure from scientific decisions.
The component boundaries are inspired by AppleCiDEr and FLARE, while the
application targets periodic white dwarf binary light curves.

| Responsibility | Component |
|---|---|
| ZTF retrieval and quality filtering | WDB photometry client using Penquins |
| Period, Fourier and eclipse measurements | WDB photometry package using periodfind and Astropy |
| Optional magnitude statistics | WDB adapter to the independent light-curve library |
| Named feature schemas and portable tables | WDB data layer |
| Workflow configuration and operation dispatch | Hyrax |
| Full-table RF/Extra Trees fitting and serialization | WDB Hyrax plugin using scikit-learn |
| Source grouping, evaluation and threshold selection | WDB scientific workflow |
| Gaia associations and astrophysical preselection | Planned application components |

## Scientific contracts

IDs, labels and catalog-truth periods are not classifier inputs. Every
observation/window of one physical source must stay in the same evaluation
group. Coordinate association must be audited before assigning groups;
different string IDs alone do not prove independence.

Median imputation and missingness indicators are fitted on training data only.
Thresholds are selected from inner out-of-fold background scores, not outer
test labels. Cross-validation must respect physical groups. Report candidates
as candidates, and document whether background labels are confirmed or provisional.

The model consumes named feature tables rather than raw light curves. Feature
versions must specify definitions, units, photometric space, bands and software
versions. Per-band tables require explicit source-level assembly. Arbitrary
statistics cannot be substituted into a fitted model's schema.

## Execution and persistence

`wdb_train` and `wdb_predict` are registered Hyrax operations for complete-table
tree models. They do not modify Hyrax or imply native distributed neural
training. Scientific preprocessing can also operate on ordinary arrays/tables.

The application saves per-source calculations and per-fit models atomically,
with single-writer locks and input/code/version checks. Bounded commands resume
compatible work. Limits are checked between sources/fits and cannot interrupt
a native calculation mid-call. Current commands run on CPU.

## Scientific development priorities

1. Assemble source-level photometry and catalog features with auditable associations.
2. Evaluate candidate feature/model configurations using disjoint physical sources.
3. Establish completeness and contamination on independent data, including hard negatives.
4. Validate any cheap preselection separately from classification, recording lost sources.
5. Freeze a scientific model and benchmark the full search before HPC deployment.

Missing catalog measurements must not automatically imply physical rejection.
No pretrained weights, real-data labels or global completeness claims are
provided by the offline example.
