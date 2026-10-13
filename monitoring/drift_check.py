"""Compare recent production inputs with the training data, measure live quality,
and decide whether retraining is needed."""

import argparse
import json
import os
import warnings

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src import tracking
from src.config import ROOT, load_params
from src.data import FEATURES
from src.metrics import score

warnings.filterwarnings("ignore")


def drift(reference: pd.DataFrame, current: pd.DataFrame, cats: list[str], html_path: str):
    definition = DataDefinition(
        numerical_columns=[c for c in FEATURES if c not in cats], categorical_columns=cats
    )
    snapshot = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current[FEATURES], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[FEATURES], data_definition=definition),
    )
    snapshot.save_html(html_path)
    share, drifted = 0.0, []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif config["type"].endswith("ValueDrift"):
            # p-value tests drift when the value is small; distance tests when it is large
            is_p_value = "p_value" in config["method"].lower()
            value, threshold = metric["value"], config["threshold"]
            if (value < threshold) if is_p_value else (value >= threshold):
                drifted.append(config["column"])
    return share, sorted(drifted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="monitoring/current.csv")
    parser.add_argument("--summary", default="monitoring/drift_summary.json")
    parser.add_argument("--report", default="monitoring/drift_report.html")
    args = parser.parse_args()

    params = load_params()
    m = params["monitor"]
    reference = pd.read_csv(ROOT / params["data"]["processed_dir"] / "train.csv")
    current = pd.read_csv(args.current)
    summary = {"production_rows": len(current)}
    if len(current) < 100:
        summary.update(status="skipped", reason="fewer than 100 predictions", retrain=False)
    else:
        share, drifted = drift(reference, current, params["model"]["cat_features"], args.report)
        labelled = current.dropna(subset=["y"])
        summary.update(
            drift_share=round(share, 3), drifted_features=drifted, labelled_rows=len(labelled)
        )
        if len(labelled) >= 50 and labelled["y"].nunique() == 2:
            live = score(
                labelled["y"], labelled["probability"], params["evaluate"]["top_k_fraction"]
            )
            summary["live"] = {k: round(v, 3) for k, v in live.items()}
        summary["retrain"] = bool(
            share >= m["drift_share_threshold"] and len(labelled) >= m["min_labelled_rows"]
        )
        summary["status"] = "drift" if share >= m["drift_share_threshold"] else "ok"

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if tracking.configure().startswith("http") and "drift_share" in summary:
        import mlflow

        mlflow.set_experiment(params["model"]["experiment"] + "-monitoring")
        with mlflow.start_run(run_name="drift-check"):
            mlflow.log_metrics(
                {
                    "drift_share": summary["drift_share"],
                    "labelled_rows": summary["labelled_rows"],
                    **{f"live_{k}": v for k, v in summary.get("live", {}).items()},
                }
            )
            mlflow.log_artifact(args.report)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"retrain={'true' if summary['retrain'] else 'false'}\n")


if __name__ == "__main__":
    main()
