"""Point MLflow at your DagsHub repository when DAGSHUB_OWNER and DAGSHUB_TOKEN are set.

Codespaces secrets and GitHub Actions both provide these two values, so every script
calls configure() before talking to MLflow. Without them, MLflow logs to a local store.
"""

import os


def configure() -> str:
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO", "mlops-bank-marketing")
    if owner and token and not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    return os.getenv("MLFLOW_TRACKING_URI", "")
