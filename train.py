"""Train, validate and export everything the dashboard needs.

Validation is strictly time-ordered (no shuffling => no leakage):
  train 0-65% | validation 65-75% (pick threshold) | test A 75-87.5% | test B 87.5-100%
The last quarter contains a NEW fraud pattern. We compare a static model with an
'adaptive' one that is retrained once labels from test A arrive.
"""
import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from generate_data import generate
from fraud_model import FEATURES, LABELS, FraudModel, engineer

REVIEW_COST = 5.0  # $ per manual review (false-positive cost)
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

raw = generate()
raw.to_csv("data/transactions.csv", index=False)
df, X = engineer(raw)
y = df["is_fraud"].to_numpy()
n = len(df)
cut = lambda p: int(n * p)
tr, va = slice(0, cut(.65)), slice(cut(.65), cut(.75))
ta, tb = slice(cut(.75), cut(.875)), slice(cut(.875), n)


def sweep(score, yt, amt, thresholds):
    rows = []
    for th in thresholds:
        flag = score >= th
        tp = int((flag & (yt == 1)).sum()); fp = int((flag & (yt == 0)).sum())
        fn = int((~flag & (yt == 1)).sum()); tn = int((~flag & (yt == 0)).sum())
        saved = float(amt[flag & (yt == 1)].sum()); missed = float(amt[~flag & (yt == 1)].sum())
        rows.append({"threshold": round(float(th), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "fpr": fp / max(fp + tn, 1), "fraud_stopped": saved, "fraud_missed": missed,
                     "review_cost": fp * REVIEW_COST, "total_cost": missed + fp * REVIEW_COST})
    return rows


def evaluate(model, sl, th):
    s = model.score(X.iloc[sl]); yt = y[sl]; amt = df["amount"].to_numpy()[sl]
    sup = model.supervised(X.iloc[sl])
    prec, rec, _ = precision_recall_curve(yt, s)
    row = sweep(s, yt, amt, [th])[0]
    # recall at a fixed 1% false-positive budget
    neg = np.sort(s[yt == 0])
    th1 = neg[int(len(neg) * 0.99)]
    r1 = float(((s >= th1) & (yt == 1)).sum() / max(yt.sum(), 1))
    return {"pr_auc": float(average_precision_score(yt, s)), "roc_auc": float(roc_auc_score(yt, s)),
            "pr_auc_supervised_only": float(average_precision_score(yt, sup)),
            "recall_at_1pct_fpr": r1, "at_threshold": row,
            "baseline_pr_auc": float(yt.mean())}, s, (prec, rec)


# ---- static model ----
static = FraudModel().fit(X.iloc[tr], y[tr])
vs = static.score(X.iloc[va])
grid = np.unique(np.quantile(vs, np.linspace(0.90, 0.9995, 120)))
vsweep = sweep(vs, y[va], df["amount"].to_numpy()[va], grid)
best = min(vsweep, key=lambda r: r["total_cost"])
TH = best["threshold"]
print(f"threshold {TH:.4f} (val cost ${best['total_cost']:.0f}, recall {best['recall']:.2f}, prec {best['precision']:.2f})")

m_val, _, _ = evaluate(static, va, TH)
m_a, s_a, pr_a = evaluate(static, ta, TH)
m_b_static, s_b_static, _ = evaluate(static, tb, TH)

# ---- adaptive model: retrain once test-A labels arrive ----
adapt = FraudModel().fit(X.iloc[: ta.stop], y[: ta.stop])
m_b_adapt, s_b_adapt, pr_b = evaluate(adapt, tb, TH)
print("test A  PR-AUC %.3f | test B static %.3f | test B adaptive %.3f" % (
    m_a["pr_auc"], m_b_static["pr_auc"], m_b_adapt["pr_auc"]))

# ---- per-pattern recall on the final period (adaptive model) ----
pat = df["pattern"].to_numpy()
patterns = {}
for p in sorted(set(pat) - {"legit"}):
    for name, m, s in (("static", static, s_b_static), ("adaptive", adapt, s_b_adapt)):
        mask = (pat[tb] == p)
        if mask.sum() == 0: continue
        patterns.setdefault(p, {"n": int(mask.sum())})[name] = float((s[mask] >= TH).mean())
# earlier-period patterns on test A
for p in sorted(set(pat) - {"legit"}):
    mask = (pat[ta] == p)
    if mask.sum():
        patterns.setdefault(p, {"n": 0}).update({"n_a": int(mask.sum()), "recall_a": float((s_a[mask] >= TH).mean())})

# ---- global importance (permutation, on test A) ----
pi = permutation_importance(static.gb, X.iloc[ta], y[ta], scoring="average_precision",
                            n_repeats=3, random_state=0)
importance = sorted([{"feature": f, "label": LABELS[f], "importance": float(v)}
                     for f, v in zip(FEATURES, pi.importances_mean)], key=lambda d: -d["importance"])

# ---- fairness / responsible-AI slice: error rates by channel & region ----
fair = []
sa = static.score(X.iloc[ta]); dta = df.iloc[ta]; yta = y[ta]
for col in ("channel", "card_home_country"):
    for g, idx in dta.groupby(col).indices.items():
        if yta[idx].sum() < 5 and (yta[idx] == 0).sum() < 50: continue
        flag = sa[idx] >= TH; neg = yta[idx] == 0
        fair.append({"slice": col, "group": g, "n": int(len(idx)),
                     "fpr": float((flag & neg).sum() / max(neg.sum(), 1)),
                     "recall": float((flag & ~neg).sum() / max((~neg).sum(), 1)) if (~neg).sum() else None})

# ---- exports ----
down = lambda p, r: [[round(float(a), 4), round(float(b), 4)] for a, b in
                     list(zip(r, p))[:: max(len(p) // 150, 1)]]
tb_df = pd.concat([df.iloc[tb], X.iloc[tb].add_prefix("f_")], axis=1); tb_df["score_static"] = s_b_static; tb_df["score_adaptive"] = s_b_adapt
stream = pd.concat([tb_df[tb_df.is_fraud == 1].sample(min(40, int(tb_df.is_fraud.sum())), random_state=1),
                    tb_df[tb_df.is_fraud == 0].sample(160, random_state=1)]).sort_values("timestamp")
stream.to_csv("artifacts/stream.csv", index=False)

metrics = {
    "dataset": {"rows": n, "fraud_rate": float(y.mean()), "days": 60,
                "train": [0, tr.stop], "val": [va.start, va.stop], "test_a": [ta.start, ta.stop],
                "test_b": [tb.start, tb.stop],
                "fraud_by_pattern": df[df.is_fraud == 1].pattern.value_counts().to_dict()},
    "threshold": TH, "review_cost": REVIEW_COST, "validation": m_val,
    "test_a": m_a, "test_b_static": m_b_static, "test_b_adaptive": m_b_adapt,
    "pr_curve_a": down(*pr_a), "pr_curve_b_adaptive": down(*pr_b),
    "sweep": sweep(s_b_adapt, y[tb], df["amount"].to_numpy()[tb], np.unique(np.quantile(s_b_adapt, np.linspace(0.9, 0.9995, 60)))),
    "patterns": patterns, "importance": importance, "fairness": fair,
    "no_model_cost_b": float(df["amount"].to_numpy()[tb][y[tb] == 1].sum()),
}
json.dump(metrics, open("artifacts/metrics.json", "w"), indent=1)
joblib.dump({"model": adapt, "threshold": TH, "baseline": X.iloc[tr].median()}, "artifacts/model.joblib")
print("done")
