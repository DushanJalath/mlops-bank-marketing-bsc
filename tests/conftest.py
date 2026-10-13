"""Shared test fixtures. Tests use small synthetic data so they run without DVC or MLflow."""

import random

import pandas as pd
import pytest

from src.data import CATEGORIES


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        row = {"age": rnd.randint(18, 90)}
        row.update({col: rnd.choice(values) for col, values in CATEGORIES.items()})
        row.update(
            {
                "campaign": rnd.randint(1, 5),
                "pdays": rnd.choice([999, 999, 999, 3, 6]),
                "previous": rnd.randint(0, 2),
                "emp.var.rate": rnd.choice([1.4, 1.1, -1.8]),
                "cons.price.idx": round(rnd.uniform(92.2, 94.8), 3),
                "cons.conf.idx": round(rnd.uniform(-50, -26), 1),
                "euribor3m": round(rnd.uniform(0.6, 5.0), 3),
                "nr.employed": rnd.choice([5228.1, 5099.1, 5191.0]),
                "y": rnd.random() < 0.3,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["y"] = df["y"].astype(int)
    return df


@pytest.fixture
def rows() -> pd.DataFrame:
    return make_rows(200)
