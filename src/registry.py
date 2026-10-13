"""Register the model from run_info.json and point the @champion alias at it.

Called by the CD workflow after a successful deployment, or by hand for the first model.
"""

import argparse
import json

import mlflow
from mlflow import MlflowClient

from src import tracking
from src.config import ROOT, load_params


def promote(alias: str, image: str | None) -> None:
    tracking.configure()
    name = load_params()["model"]["name"]
    run = json.loads((ROOT / "run_info.json").read_text())
    client = MlflowClient()

    # Re-use the version if this run was already registered (e.g. a re-run of the workflow).
    existing = [
        v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run["run_id"]
    ]
    if existing:
        version = existing[0].version
    else:
        version = mlflow.register_model(run["model_uri"], name).version

    if image:
        client.set_model_version_tag(name, version, "image", image)
    client.set_registered_model_alias(name, alias, version)
    print(f"{name} version {version} is now @{alias}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["promote"])
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--image", default=None, help="container image that serves this version")
    args = parser.parse_args()
    promote(args.alias, args.image)
