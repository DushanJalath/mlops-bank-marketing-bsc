import json

from monitoring.export import parse_events
from tests.conftest import make_rows


def test_parse_events_joins_feedback():
    row = make_rows(1).drop(columns=["y"]).iloc[0].to_dict()
    lines = [
        'INFO:     127.0.0.1 - "POST /predict HTTP/1.1" 200 OK',  # non-JSON lines are skipped
        json.dumps(
            {
                "event": "prediction",
                "request_id": "a",
                "probability": 0.3,
                "prediction": 0,
                "model_version": "v1",
                "timestamp": "2026-01-01T00:00:01",
                "features": row,
            }
        ),
        json.dumps(
            {
                "event": "prediction",
                "request_id": "b",
                "probability": 0.7,
                "prediction": 1,
                "model_version": "v1",
                "timestamp": "2026-01-01T00:00:02",
                "features": row,
            }
        ),
        json.dumps({"event": "feedback", "request_id": "a", "actual_outcome": 1}),
    ]
    df = parse_events(lines)
    assert list(df["request_id"]) == ["a", "b"]
    assert df.loc[0, "y"] == 1
    assert df["y"].isna().sum() == 1  # b has no feedback yet
