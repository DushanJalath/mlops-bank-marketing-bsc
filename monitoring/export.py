"""Pull prediction and feedback events from Log Analytics and turn them into a table.

  drift window:   python -m monitoring.export --hours 6 --out monitoring/current.csv
  retraining:     python -m monitoring.export --hours 720 --labelled-only \
                      --append-to data/production/labelled.csv
Needs LOG_ANALYTICS_WORKSPACE_ID and an Azure CLI login (az login, or azure/login in Actions).
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.data import FEATURES

QUERY = """
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "{app}"
| where Log_s startswith "{{"
| project TimeGenerated, Log_s
"""


def fetch_rows(workspace_id: str, app: str, hours: int) -> list[str]:
    from azure.identity import AzureCliCredential
    from azure.monitor.query import LogsQueryClient, LogsQueryStatus

    client = LogsQueryClient(AzureCliCredential())
    result = client.query_workspace(
        workspace_id, QUERY.format(app=app), timespan=timedelta(hours=hours)
    )
    if result.status != LogsQueryStatus.SUCCESS:
        raise RuntimeError(f"Log Analytics query failed: {result}")
    table = result.tables[0]
    column = list(table.columns).index("Log_s")
    return [row[column] for row in table.rows]


def parse_events(lines: list[str]) -> pd.DataFrame:
    """One row per prediction, with the real outcome in column y when feedback arrived."""
    predictions, feedback = [], {}
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue  # uvicorn access logs and other non-JSON lines
        if event.get("event") == "prediction":
            predictions.append(
                {
                    **event["features"],
                    "probability": event["probability"],
                    "model_version": event["model_version"],
                    "request_id": event["request_id"],
                    "timestamp": event["timestamp"],
                }
            )
        elif event.get("event") == "feedback":
            feedback[event["request_id"]] = event["actual_outcome"]
    df = pd.DataFrame(
        predictions, columns=FEATURES + ["probability", "model_version", "request_id", "timestamp"]
    )
    df["y"] = df["request_id"].map(feedback)
    return df.drop_duplicates("request_id").sort_values("timestamp").reset_index(drop=True)


def append_labelled(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    labelled = new.dropna(subset=["y"]).astype({"y": int})
    keep = FEATURES + ["y", "timestamp", "request_id"]
    combined = (
        pd.concat([pd.read_csv(path), labelled[keep]], ignore_index=True)
        if path.exists()
        else labelled[keep]
    )
    combined = combined.drop_duplicates("request_id").sort_values("timestamp")
    combined.to_csv(path, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--out", help="write all predictions in the window to this CSV")
    parser.add_argument("--labelled-only", action="store_true")
    parser.add_argument("--append-to", help="merge labelled rows into this CSV")
    args = parser.parse_args()

    workspace = os.environ["LOG_ANALYTICS_WORKSPACE_ID"]
    app = os.getenv("CONTAINER_APP_NAME", "ca-bank-marketing")
    df = parse_events(fetch_rows(workspace, app, args.hours))
    print(f"{len(df)} predictions, {df['y'].notna().sum()} with feedback")
    if args.labelled_only:
        df = df.dropna(subset=["y"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    if args.append_to:
        total = append_labelled(df, Path(args.append_to))
        print(f"{args.append_to} now has {len(total)} labelled rows")


if __name__ == "__main__":
    main()
