"""FraudShield API + dashboard.  Run: python app.py  ->  http://localhost:5050"""
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from fraud_model import MERCHANT_RISK, CATS, FEATURES, row_features  # noqa: F401

app = Flask(__name__, static_folder="static")
bundle = joblib.load("artifacts/model.joblib")
MODEL, TH, BASE = bundle["model"], bundle["threshold"], bundle["baseline"]
METRICS = json.load(open("artifacts/metrics.json"))
STREAM = pd.read_csv("artifacts/stream.csv")


def to_features(t: dict) -> pd.DataFrame:
    """Build one feature row from a user-entered transaction."""
    df = pd.DataFrame([{
        "amount": float(t.get("amount", 50)), "merchant_category": t.get("merchant_category", "grocery"),
        "channel": t.get("channel", "pos"), "country": t.get("country", "US"),
        "card_home_country": t.get("card_home_country", "US"), "hour": int(t.get("hour", 12)),
        "new_device": int(bool(t.get("new_device", 0))), "account_age_days": int(t.get("account_age_days", 800)),
    }])
    f = row_features(df)
    avg = max(float(t.get("card_avg_spend", 40)), 1)
    f["log_amount"] = pd.Series([__import__("math").log1p(df.amount[0])])
    f["amount_vs_card_avg"] = df.amount[0] / avg
    f["txn_1h"] = float(t.get("txn_1h", 0)); f["txn_24h"] = float(t.get("txn_24h", 0))
    return f[FEATURES]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/api/score")
def score():
    X = to_features(request.get_json(force=True))
    s, why = MODEL.explain(X, BASE)
    decision = "block" if s >= TH else ("review" if s >= TH * 0.6 else "approve")
    return jsonify({"score": s, "threshold": TH, "decision": decision, "reasons": why})


@app.get("/api/stream")
def stream():
    rows = STREAM.to_dict("records")
    return jsonify([{k: r[k] for k in ("txn_id", "amount", "merchant_category", "channel", "country",
                                       "card_home_country", "hour", "is_fraud", "pattern",
                                       "score_static", "score_adaptive")} for r in rows])


if __name__ == "__main__":
    app.run(port=5050, debug=False)
