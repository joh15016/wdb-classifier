"""Source-grouped folds and training-only operating-point selection."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def splits(frame, n_splits, seed):
    labels = frame.label.to_numpy()
    if frame.group_id.is_unique:
        if frame.label.value_counts().min() < n_splits:
            raise ValueError("Too few sources of one class for the requested folds")
        return list(
            StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(frame)), labels)
        )
    by_group = frame.drop_duplicates("group_id")
    if by_group.label.value_counts().min() < n_splits:
        raise ValueError("Too few physical groups of one class for the requested folds")
    return list(
        StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(frame)), labels, frame.group_id
        )
    )


def outer_splits(frame, n_splits, seed, assignments=None):
    if assignments is None:
        result = splits(frame, n_splits, seed)
    else:
        if (
            not {"source_id", "fold"}.issubset(assignments)
            or not assignments.source_id.is_unique
            or set(assignments.source_id.astype(str)) != set(frame.source_id)
        ):
            raise ValueError("Fold assignments must cover the exact source set once")
        folds = (
            assignments.assign(source_id=assignments.source_id.astype(str))
            .set_index("source_id")
            .loc[frame.source_id, "fold"]
        )
        if not folds.isin(range(n_splits)).all() or set(folds) != set(range(n_splits)):
            raise ValueError("Fold numbers must cover 0 through n_splits - 1")
        result = [
            (np.flatnonzero(folds.to_numpy() != k), np.flatnonzero(folds.to_numpy() == k))
            for k in range(n_splits)
        ]
    for train, test in result:
        if set(frame.group_id.iloc[train]).intersection(frame.group_id.iloc[test]):
            raise ValueError("Physical-source group crosses training and test folds")
        if frame.label.iloc[train].nunique() != 2 or frame.label.iloc[test].nunique() != 2:
            raise ValueError("Every training and test fold must contain both classes")
    return result


def threshold(labels, scores, budget):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if not 0 < budget < 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Finite aligned validation scores and 0 < budget < 1 are required")
    background = scores[labels == 0]
    if not len(background):
        raise ValueError("Validation background sources are required")
    return float(np.nextafter(np.quantile(background, 1 - budget, method="higher"), np.inf))


def metrics(labels, scores, flagged):
    tn, fp, fn, tp = confusion_matrix(labels, flagged, labels=[0, 1]).ravel()
    return dict(
        n=int(len(labels)),
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        tn=int(tn),
        recall=float(tp / (tp + fn)) if tp + fn else None,
        background_flag_rate=float(fp / (fp + tn)) if fp + tn else None,
        average_precision=float(average_precision_score(labels, scores)),
        roc_auc=float(roc_auc_score(labels, scores)),
    )
