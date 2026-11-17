# Validation and scientific interpretation

The automated suite checks schema ordering, protected metadata, training-only
preprocessing, source-group isolation, threshold ties, Hyrax operation dispatch,
checkpoint reuse, saved inference, credential handling and feature extraction.
All fixtures are generated locally; the tests do not query real survey services.

The offline example uses 96 toy training rows and 12 disjoint toy prediction
rows. It exercises nine small tree fits and repeat-run checkpoint reuse. It
does not simulate physical binary populations or establish scientific recovery.

Feature tests verify a small synthetic period search and cache replay. This
checks software consistency, not whether every selected period is a true orbital
period. Eclipse-support requirements reduce single-point artifacts but do not
guarantee physical binary identification.

Before reporting WDB completeness, contamination or discoveries:

- Freeze the target population, photometric quality rules and feature schema.
- Audit source associations and keep each physical source in one evaluation group.
- Define provisional versus verified labels and include challenging negative sources.
- Separate model/threshold selection from independent evaluation.
- Evaluate losses introduced by any preselection before classification.
- Measure candidate yield and contamination at the intended survey prevalence.
- Benchmark extraction and inference costs on representative observations.

Classifier scores and threshold budgets are not calibrated population WDB
probabilities or survey purity guarantees. A positive output is a candidate
requiring further examination. Gaia associations, calibrated ensembles and
complete automated search orchestration remain development tasks.
