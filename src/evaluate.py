"""Score the trained model on the evaluation set, explain it with SHAP, log both to MLflow."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402

from src import tracking
from src.config import ROOT, load_params  # noqa: E402
from src.metrics import score  # noqa: E402


def main() -> None:
    tracking.configure()
    params = load_params()
    k = params["evaluate"]["top_k_fraction"]
    df = pd.read_csv(ROOT / params["data"]["processed_dir"] / "eval.csv")
    X, y = df.drop(columns=["y"]), df["y"]

    model = CatBoostClassifier()
    model.load_model(str(ROOT / "models" / "model.cbm"))
    metrics = score(y, model.predict_proba(X)[:, 1], k)
    metrics["eval_rows"] = len(df)
    (ROOT / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    # SHAP: how much each feature pushed each prediction up or down.
    sample = X.sample(min(500, len(X)), random_state=0)
    pool = Pool(sample, cat_features=params["model"]["cat_features"])
    shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    plot_path = ROOT / "plots" / "shap_summary.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close("all")

    run_id = json.loads((ROOT / "run_info.json").read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{name}": value for name, value in metrics.items()})
        mlflow.log_artifact(str(plot_path))


if __name__ == "__main__":
    main()
