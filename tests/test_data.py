import pandas as pd
import pandera.errors
import pytest

from src.config import load_params
from src.data import FEATURES, build_sets, validate


def test_contract_accepts_valid_rows(rows):
    assert len(validate(rows)) == len(rows)


def test_contract_rejects_bad_pdays(rows):
    rows.loc[0, "pdays"] = -1  # the upstream bug from the incident drill
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_contract_rejects_unknown_category(rows):
    rows.loc[0, "job"] = "astronaut"
    with pytest.raises(pandera.errors.SchemaError):
        validate(rows)


def test_leakage_column_is_never_a_feature():
    for column in load_params()["data"]["drop_columns"]:
        assert column not in FEATURES


def test_first_model_uses_history_only(rows):
    p = {"history_frac": 0.8, "val_frac": 0.2, "holdout_frac": 0.3, "seed": 1}
    train, evaluation, future = build_sets(rows, pd.DataFrame(), p)
    assert len(future) == 40  # newest 20% is kept for the replayer
    assert len(train) + len(evaluation) == 160
    assert set(train.index).isdisjoint(evaluation.index)


def test_retraining_evaluates_on_newest_production_rows(rows):
    p = {"history_frac": 0.8, "val_frac": 0.2, "holdout_frac": 0.3, "seed": 1}
    production = validate(rows.head(100))
    train, evaluation, _ = build_sets(rows, production, p)
    assert len(evaluation) == 30  # newest 30% of production rows
    assert len(train) == 160 + 70  # all history plus the older production rows
