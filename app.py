"""FraudShield API + dashboard.  Run: python app.py  ->  http://localhost:5050"""
import json
import math
import os
import threading
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from fraud_model import CATS, FEATURES, row_features

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
STATIC = os.path.join(HERE, "static")

app = Flask(__name__, static_folder=STATIC)
bundle = joblib.load(os.path.join(ART, "model.joblib"))
MODEL, TH, BASE = bundle["model"], bundle["threshold"], bundle["baseline"]
METRICS = json.load(open(os.path.join(ART, "metrics.json")))
STREAM = pd.read_csv(os.path.join(ART, "stream.csv"))
BY_ID = {r["txn_id"]: r for r in STREAM.to_dict("records")}
# Serverless hosts (Vercel) only allow writes under /tmp, so verdicts live there and reset on cold starts.
REVIEW_FILE = os.environ.get("REVIEW_FILE") or ("/tmp/reviews.json" if os.environ.get("VERCEL") else os.path.join(ART, "reviews.json"))
_lock = threading.Lock()
COUNTRIES = ["US", "GB", "DE", "IN", "BR", "NG", "RU", "CN"]


def load_reviews():
    try:
        return json.load(open(REVIEW_FILE))
    except (OSError, ValueError):
        return {}


REVIEWS = load_reviews()


def decide(score: float) -> str:
    return "block" if score >= TH else ("review" if score >= TH * 0.6 else "approve")


def validate(t: dict):
    """Return (clean dict, errors by field). Errors use the form's own field names."""
    errs, out = {}, {}

    def num(key, lo, hi, default, integer=False):
        v = t.get(key, default)
        try:
            v = float(v)
            if math.isnan(v) or v < lo or v > hi or (integer and v != int(v)):
                raise ValueError
        except (TypeError, ValueError):
            errs[key] = f"Enter a {'whole ' if integer else ''}number from {lo:,} to {hi:,}."
            return default
        return int(v) if integer else v

    out["amount"] = num("amount", 0.01, 1_000_000, 50)
    out["card_avg_spend"] = num("card_avg_spend", 0.01, 1_000_000, 40)
    out["hour"] = num("hour", 0, 23, 12, True)
    out["account_age_days"] = num("account_age_days", 0, 36500, 800, True)
    out["txn_1h"] = num("txn_1h", 0, 500, 0, True)
    out["txn_24h"] = num("txn_24h", 0, 5000, 0, True)
    if "txn_1h" not in errs and "txn_24h" not in errs and out["txn_24h"] < out["txn_1h"]:
        errs["txn_24h"] = "Can't be fewer than the payments in the last hour."
    for key, allowed in (("merchant_category", CATS["merchant_category"]), ("channel", CATS["channel"]),
                         ("country", COUNTRIES), ("card_home_country", COUNTRIES)):
        v = t.get(key)
        if v not in allowed:
            errs[key] = "Choose an option from the list."
        out[key] = v if v in allowed else allowed[0]
    out["new_device"] = int(bool(t.get("new_device", False)))
    return out, errs


def to_features(t: dict) -> pd.DataFrame:
    df = pd.DataFrame([{k: t[k] for k in ("amount", "merchant_category", "channel", "country",
                                          "card_home_country", "hour", "new_device", "account_age_days")}])
    f = row_features(df)
    f["log_amount"] = math.log1p(t["amount"])
    f["amount_vs_card_avg"] = t["amount"] / max(t["card_avg_spend"], 1)
    f["txn_1h"], f["txn_24h"] = float(t["txn_1h"]), float(t["txn_24h"])
    return f[FEATURES]


def public_row(r: dict) -> dict:
    keys = ("txn_id", "amount", "merchant_category", "channel", "country", "card_home_country", "hour",
            "new_device", "account_age_days", "is_fraud", "pattern", "score_static", "score_adaptive")
    out = {k: r[k] for k in keys}
    out["review"] = REVIEWS.get(r["txn_id"])
    return out


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/api/score")
def score():
    clean, errs = validate(request.get_json(silent=True) or {})
    if errs:
        return jsonify({"error": "Some fields need fixing.", "fields": errs}), 400
    s, why = MODEL.explain(to_features(clean), BASE)
    return jsonify({"score": s, "threshold": TH, "decision": decide(s), "reasons": why})


@app.get("/api/stream")
def stream():
    return jsonify([public_row(r) for r in BY_ID.values()])


@app.get("/api/explain/<txn_id>")
def explain(txn_id):
    r = BY_ID.get(txn_id)
    if r is None:
        return jsonify({"error": f"No transaction {txn_id} in this sample."}), 404
    X = pd.DataFrame([{f: r["f_" + f] for f in FEATURES}])
    s, why = MODEL.explain(X, BASE)
    return jsonify({**public_row(r), "score": s, "threshold": TH, "decision": decide(s), "reasons": why})


@app.post("/api/review/<txn_id>")
def review(txn_id):
    if txn_id not in BY_ID:
        return jsonify({"error": f"No transaction {txn_id} in this sample."}), 404
    verdict = (request.get_json(silent=True) or {}).get("verdict")
    if verdict not in ("fraud", "legit", None):
        return jsonify({"error": "Verdict must be 'fraud' or 'legit'."}), 400
    with _lock:
        if verdict is None:
            REVIEWS.pop(txn_id, None)
        else:
            REVIEWS[txn_id] = {"verdict": verdict, "at": int(time.time())}
        tmp = REVIEW_FILE + ".tmp"
        json.dump(REVIEWS, open(tmp, "w"))
        os.replace(tmp, REVIEW_FILE)
    return jsonify(public_row(BY_ID[txn_id]))


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(STATIC, "index.html")


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5050)), debug=False)
