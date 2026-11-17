import joblib
import numpy as np
import pandas as pd
import pytest

from wdb_classifier.artifacts import write_json
from wdb_classifier.config import TrainingConfig
from wdb_classifier.data import FeatureSchema, load_features, validate_features
from wdb_classifier.evaluation import outer_splits, splits, threshold
from wdb_classifier.models import fit_tree, predict_score


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(17)
    labels = np.array([0, 1] * 30)
    values = rng.normal(size=(60, 3))
    values[:, 0] += labels * 3
    values[:10, 2] = np.nan
    frame = pd.DataFrame(values, columns=["a", "b", "c"])
    frame["source_id"] = [f"source-{i}" for i in range(60)]
    frame["label"] = labels
    schema = FeatureSchema("test_v1", ("a", "b", "c"))
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    write_json(tmp_path / "schema.json", schema.as_dict())
    config = TrainingConfig(
        str(path),
        str(tmp_path / "schema.json"),
        str(tmp_path / "run"),
        outer_folds=3,
        inner_folds=2,
        parameters=dict(n_estimators=8),
    )
    return frame, values, schema, config


def test_identifiers_and_labels_cannot_be_features():
    for names in (("label",), ("source_id",), ("known_period_days",), ("a", "a")):
        with pytest.raises(ValueError):
            FeatureSchema("bad", names)


def test_schema_controls_order_and_rejects_duplicate_sources(inputs):
    frame, values, schema, _ = inputs
    _, actual = validate_features(frame[list(reversed(frame.columns))], schema, labeled=True)
    np.testing.assert_array_equal(actual, values.astype(np.float32).astype(float))
    with pytest.raises(ValueError, match="unique"):
        validate_features(pd.concat([frame, frame.iloc[:1]]), schema, labeled=True)


def test_preprocessing_is_training_only_and_predictions_ignore_label(inputs):
    frame, values, schema, _ = inputs
    state = fit_tree(values[:40], frame.label.iloc[:40], schema, seed=12, parameters=dict(n_estimators=8))
    np.testing.assert_array_equal(
        state["medians"], np.nanmedian(values[:40].astype(np.float32).astype(float), axis=0)
    )
    frame["label"] = 1 - frame.label
    _, x = validate_features(frame, schema)
    np.testing.assert_array_equal(predict_score(state, x[40:]), predict_score(state, values[40:]))


def test_physical_groups_are_never_split_between_roles(inputs):
    frame, _, schema, _ = inputs
    frame["group_id"] = [f"g-{i // 2}" for i in range(60)]
    frame["label"] = [(i // 2) % 2 for i in range(60)]
    frame, _ = validate_features(frame, schema, labeled=True)
    for a, b in splits(frame, 3, 12):
        assert not set(frame.group_id.iloc[a]).intersection(frame.group_id.iloc[b])
    assignment = pd.DataFrame(dict(source_id=frame.source_id, fold=np.arange(60) % 3))
    with pytest.raises(ValueError, match="group crosses"):
        outer_splits(frame, 3, 12, assignment)


def test_background_threshold_excludes_quantile_ties():
    y, score = np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.2, 0.9])
    assert threshold(y, score, 0.1) == np.nextafter(0.2, np.inf)
    score[-1] = 0.01
    assert threshold(y, score, 0.1) == np.nextafter(0.2, np.inf)


def test_real_hyrax_dispatch_checkpoint_resume_and_prediction(inputs, tmp_path):
    from wdb_classifier.hyrax_plugin import create_workflow

    frame, _, schema, config = inputs
    workflow = create_workflow(config)
    first = workflow.wdb_train(max_new_fits=1)
    assert first["engine"] == "Hyrax" and not first["complete"] and first["new_fits"] == 1
    path = tmp_path / "run/folds/0/inner0/model.joblib"
    before = path.read_bytes(), path.stat().st_mtime_ns
    second = workflow.wdb_train()
    assert second["complete"] and second["new_fits"] == 8
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    assert workflow.wdb_train(max_new_fits=0)["new_fits"] == 0
    predicted = pd.read_csv(tmp_path / "run/oof_predictions.csv")
    assert len(predicted) == 120 and predicted.groupby("budget").source_id.nunique().eq(60).all()
    with pytest.raises(ValueError, match="overlap training"):
        workflow.wdb_predict(run=config.output, features=config.features, output=tmp_path / "blocked.csv")
    unseen = frame.drop(columns="label").assign(source_id=[f"new-{i}" for i in range(len(frame))])
    unseen.to_parquet(tmp_path / "unseen.parquet", index=False)
    output = tmp_path / "predictions.parquet"
    result = workflow.wdb_predict(run=config.output, features=tmp_path / "unseen.parquet", output=output)
    assert result["sources"] == 60
    actual = pd.read_parquet(output)
    assert not actual.training_overlap.any() and set(actual.label_name) <= {"WDB candidate", "Not flagged"}
    state = joblib.load(tmp_path / "run/folds/0/bundle.joblib")
    _, x = load_features(tmp_path / "unseen.parquet", schema)
    assert np.isfinite(predict_score(state, x)).all()
    with pytest.raises(FileExistsError):
        workflow.wdb_predict(run=config.output, features=tmp_path / "unseen.parquet", output=output)


def test_changed_inputs_cannot_reuse_checkpoint(inputs):
    from wdb_classifier.training import train

    frame, _, _, config = inputs
    train(config, max_new_fits=0)
    frame.loc[0, "a"] += 1
    frame.to_parquet(config.features, index=False)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        train(config, max_new_fits=0)


def test_feature_provider_is_lazy_and_has_separate_schema():
    from wdb_classifier.features import extract_statistics

    pytest.importorskip("light_curve")
    photometry = pd.DataFrame(
        dict(
            source_id=["test"] * 8,
            band=["g"] * 8,
            time=[5.0, 2.0, 2.0, 3.0, 4.0, 6.0, 7.0, np.nan],
            mag=[18.0, 18.0, 19.0, 20.0, 18.0, 18.0, 18.0, 18.0],
            mag_err=[0.1] * 8,
        )
    )
    result, manifest = extract_statistics(photometry)
    assert manifest["baseline"] == "all supplied epochs" and manifest["input_space"] == "magnitude"
    assert result.lc_mag_v1_n_valid.iloc[0] == 6
    assert result.lc_mag_v1_n_invalid.iloc[0] == 1
    assert set(manifest["columns"]).issubset(result)
    assert result.source_id.tolist() == ["test"]
