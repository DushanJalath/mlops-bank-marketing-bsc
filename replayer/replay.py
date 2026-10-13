"""Simulate production traffic against the live API, then send the real outcomes as feedback.

Modes:
  normal     random rows from the training period: what "no drift" looks like
  future     the newest rows of the dataset, in date order: real drift
  pdays-bug  like future, but an upstream system now sends pdays=-1 instead of 999

Example:
  uv run python replayer/replay.py https://<app-url> --mode future --rows 3000
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_rows(mode: str, rows: int, start: int) -> pd.DataFrame:
    processed = ROOT / "data" / "processed"
    if mode == "normal":
        df = pd.read_csv(processed / "train.csv").sample(rows, random_state=7)
    else:
        df = pd.read_csv(processed / "future.csv").iloc[start : start + rows]
    if mode == "pdays-bug":
        df = df.assign(pdays=df["pdays"].replace(999, -1))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="base URL of the API, without a trailing slash")
    parser.add_argument("--mode", choices=["normal", "future", "pdays-bug"], default="future")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--start", type=int, default=0, help="first future row to send")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--feedback-delay",
        type=float,
        default=20,
        help="seconds to wait before sending the real outcomes",
    )
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    df = load_rows(args.mode, args.rows, args.start)
    outcomes = df["y"].astype(int).tolist()
    payloads = df.drop(columns=["y"]).to_dict(orient="records")
    client = httpx.Client(base_url=args.url, timeout=30)

    def send(i: int):
        response = client.post("/predict", json=payloads[i])
        return i, response.status_code, response.json() if response.status_code == 200 else None

    print(f"Sending {len(payloads)} '{args.mode}' requests to {args.url} ...")
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(send, range(len(payloads))))
    ok = [(i, body) for i, status, body in results if status == 200]
    rejected = sum(1 for _, status, _ in results if status == 422)
    failed = len(results) - len(ok) - rejected
    print(
        f"done in {time.time() - started:.0f}s: {len(ok)} ok, {rejected} rejected by the "
        f"contract (422), {failed} other errors"
    )
    if ok:
        mean_probability = sum(body["probability"] for _, body in ok) / len(ok)
        print(
            f"mean predicted probability {mean_probability:.3f}, "
            f"actual subscribe rate {sum(outcomes[i] for i, _ in ok) / len(ok):.3f}"
        )

    if args.no_feedback or not ok:
        return
    print(f"Waiting {args.feedback_delay:.0f}s before sending feedback (outcomes arrive later)...")
    time.sleep(args.feedback_delay)

    def send_feedback(item):
        i, body = item
        client.post(
            "/feedback", json={"request_id": body["request_id"], "actual_outcome": outcomes[i]}
        )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(send_feedback, ok))
    print(f"Sent {len(ok)} feedback events.")
    sys.exit(0)


if __name__ == "__main__":
    main()
