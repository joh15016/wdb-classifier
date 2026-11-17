# Credits and dependency notices

## Inspiration and contributor credit

The application's separation of retrieval, feature generation and machine
learning is inspired by [AppleCiDEr](https://github.com/applecider-ml/applecider)
and [FLARE](https://github.com/applecider-ml/flare). Neither application nor its
pretrained classifiers are included or required at runtime. The independent
`light-curve` statistics library is not owned by FLARE.

Scientific period/Fourier and eclipse routines include contributions credited
to `ManiacUrgency` and the WDB research contributors. Preserve contributor
attribution; complete authorship and authorization for a public source license
must be confirmed by the project owners. Period peak and alias-family methods
also draw on SCoPe's published software methods and the periodfind ecosystem.

## Separately installed libraries

The following reported licenses describe the selected dependencies, not a
license for this application or an exhaustive transitive-dependency audit.
The libraries are installed separately and are not vendored into this source.

| Library | Role | Reported license | Source |
|---|---|---|---|
| Hyrax | Workflow integration | MIT | https://github.com/lincc-frameworks/hyrax |
| scikit-learn | Tree models | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| Penquins | ZTF database access | MIT | https://github.com/skyportal/penquins |
| periodfind / periodfind_cpu | Periodograms and Fourier fits | BSD-3-Clause | https://github.com/scope-ml/periodfind |
| light-curve, optional | Magnitude statistics | GPL-3.0-or-later | https://github.com/light-curve/light-curve-python |

Dependencies retain their own licenses and citation requirements. Optionality
alone does not determine distribution obligations. Review the intended source
and binary distribution with the group/institution when selecting a license.

No license has yet been assigned to this project's own source. Do not invent
copyright holders, remove contributor credit or describe it as permissively
licensed before the owners and contributors have agreed.
