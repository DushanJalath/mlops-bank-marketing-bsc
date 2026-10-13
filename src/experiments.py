"""Module 3: compare three model families on the same data, all tracked in MLflow.

This is exploration, separate from the production pipeline in dvc.yaml.
"""

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import tracking
from src.config import ROOT, load_params
from src.metrics import score


def build_models(cat_features: list[str], numeric: list[str]) -> dict:
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), numeric),
        ]
    )
    return {
        "logistic_regression": make_pipeline(encode, LogisticRegression(max_iter=1000)),
        "catboost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            cat_features=cat_features,
            verbose=0,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
        ),
    }


def main() -> None:
    tracking.configure()
    params = load_params()
    cats = params["model"]["cat_features"]
    k = params["evaluate"]["top_k_fraction"]
    data_dir = ROOT / params["data"]["processed_dir"]
    train, val = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "eval.csv")
    X_tr, y_tr = train.drop(columns=["y"]), train["y"]
    X_val, y_val = val.drop(columns=["y"]), val["y"]
    numeric = [c for c in X_tr.columns if c not in cats]

    mlflow.set_experiment(params["model"]["experiment"] + "-experiments")
    results = []
    for name, model in build_models(cats, numeric).items():
        X_fit, X_eval = X_tr, X_val
        if name == "xgboost":  # XGBoost needs pandas 'category' columns
            X_fit = X_tr.astype({c: "category" for c in cats})
            X_eval = X_val.astype({c: pd.CategoricalDtype(X_fit[c].cat.categories) for c in cats})
        with mlflow.start_run(run_name=name):
            model.fit(X_fit, y_tr)
            metrics = score(y_val, model.predict_proba(X_eval)[:, 1], k)
            mlflow.log_param("model_family", name)
            mlflow.log_metrics(metrics)
        results.append({"model": name, **metrics})

    table = pd.DataFrame(results).set_index("model").round(3)
    print(table[["pr_auc", "roc_auc", "recall_top_k"]].sort_values("pr_auc", ascending=False))


if __name__ == "__main__":
    main()
