from src.config import ROOT, load_params


def test_params_file_exists():
    assert (ROOT / "params.yaml").exists()


def test_duration_is_dropped():
    assert "duration" in load_params()["data"]["drop_columns"]


def test_split_fractions_are_sensible():
    data = load_params()["data"]
    assert 0 < data["history_frac"] < 1
    assert 0 < data["val_frac"] < 1
