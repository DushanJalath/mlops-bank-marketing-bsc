"""Validate the raw data against a contract, then build the training and evaluation sets."""

import pandas as pd
import pandera.pandas as pa

from src.config import ROOT, load_params

CATEGORIES = {
    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": [
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"],
}

# The data contract: every feature the model uses, with its type and allowed values.
FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "age": pa.Column(int, pa.Check.in_range(17, 100)),
        **{col: pa.Column(str, pa.Check.isin(values)) for col, values in CATEGORIES.items()},
        "campaign": pa.Column(int, pa.Check.ge(1)),
        "pdays": pa.Column(int, pa.Check.in_range(0, 999)),  # 999 = never contacted before
        "previous": pa.Column(int, pa.Check.ge(0)),
        "emp.var.rate": pa.Column(float),
        "cons.price.idx": pa.Column(float),
        "cons.conf.idx": pa.Column(float),
        "euribor3m": pa.Column(float),
        "nr.employed": pa.Column(float),
    },
    coerce=True,
)
FEATURES = list(FEATURE_SCHEMA.columns)


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if not set(df["y"].unique()) <= {"yes", "no"}:
        raise ValueError("target column y must only contain 'yes' or 'no'")
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the features against the contract. Raises pandera.errors.SchemaError on bad data."""
    checked = FEATURE_SCHEMA.validate(df[FEATURES])
    return checked.assign(y=df["y"].astype(int).to_numpy())


def split_history(df: pd.DataFrame, history_frac: float, val_frac: float, seed: int):
    """Rows are in date order. The oldest part is history; the newest part is the 'future'
    the replayer sends to production. History is split randomly into train and validation."""
    cut = int(len(df) * history_frac)
    history, future = df.iloc[:cut], df.iloc[cut:]
    val = history.groupby("y", group_keys=False).sample(frac=val_frac, random_state=seed)
    train = history.drop(val.index)
    return history, train, val, future


def build_sets(raw: pd.DataFrame, production: pd.DataFrame, p: dict):
    history, train, val, future = split_history(raw, p["history_frac"], p["val_frac"], p["seed"])
    if len(production) == 0:
        # First model: train on history, judge it on the held-out validation rows.
        return train, val, future
    # Retraining: the newest labelled production rows become the evaluation set,
    # and everything older (all history + older production rows) is used for training.
    production = production.reset_index(drop=True)
    cut = int(len(production) * (1 - p["holdout_frac"]))
    older, holdout = production.iloc[:cut], production.iloc[cut:]
    return pd.concat([history, older], ignore_index=True), holdout, future


def main() -> None:
    p = load_params()["data"]
    raw = validate(load_raw(ROOT / p["raw"]))
    production = pd.read_csv(ROOT / p["production"])
    if len(production):
        production = validate(production.sort_values("timestamp"))
    leaked = set(p["drop_columns"]) & set(raw.columns)
    if leaked:  # the contract only keeps known features, so this should never happen
        raise ValueError(f"leakage columns reached the model data: {leaked}")

    train, evaluation, future = build_sets(raw, production, p)
    out = ROOT / p["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    evaluation.to_csv(out / "eval.csv", index=False)
    future.to_csv(out / "future.csv", index=False)
    print(
        f"train={len(train)}  eval={len(evaluation)}  future={len(future)}  "
        f"production rows used={len(production)}"
    )
    print(f"positive rate: train={train.y.mean():.3f}  eval={evaluation.y.mean():.3f}")


if __name__ == "__main__":
    main()
