"""Feature engineering + model definition shared by training and the API."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

MERCHANT_RISK = {"gift_cards": 3.0, "crypto": 3.5, "electronics": 1.8, "travel": 1.4}
CATS = {
    "merchant_category": ["grocery", "fuel", "restaurant", "travel", "electronics",
                          "gift_cards", "crypto", "online_retail", "pharmacy", "utilities"],
    "channel": ["pos", "online", "mobile", "atm"],
}
FEATURES = ["log_amount", "amount_vs_card_avg", "txn_1h", "txn_24h", "foreign",
            "hour", "night", "new_device", "account_age_days", "merchant_risk",
            "merchant_code", "channel_code", "tiny_amount"]
LABELS = {
    "log_amount": "Transaction amount", "amount_vs_card_avg": "Amount vs. card's usual spend",
    "txn_1h": "Card activity (last hour)", "txn_24h": "Card activity (last 24h)",
    "foreign": "Foreign-country payment", "hour": "Hour of day", "night": "Night-time (1-5am)",
    "new_device": "Unrecognised device", "account_age_days": "Account age",
    "merchant_risk": "Merchant risk level", "merchant_code": "Merchant category",
    "channel_code": "Payment channel", "tiny_amount": "Micro-amount (card testing)",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features only: every value uses information available at txn time."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    ts = df["timestamp"].to_numpy()
    txn_1h = np.zeros(len(df))
    txn_24h = np.zeros(len(df))
    avg_prev = np.zeros(len(df))
    for _, idx in df.groupby("card_id").indices.items():
        t = ts[idx]
        txn_1h[idx] = np.arange(len(idx)) - np.searchsorted(t, t - 3600, side="left")
        txn_24h[idx] = np.arange(len(idx)) - np.searchsorted(t, t - 86400, side="left")
        amt = df["amount"].to_numpy()[idx]
        csum = np.cumsum(amt) - amt
        cnt = np.arange(len(idx))
        avg_prev[idx] = np.where(cnt > 0, csum / np.maximum(cnt, 1), np.nan)
    overall = df["amount"].median()
    avg_prev = np.where(np.isnan(avg_prev), overall, avg_prev)
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount"])
    out["amount_vs_card_avg"] = df["amount"] / np.maximum(avg_prev, 1)
    out["txn_1h"], out["txn_24h"] = txn_1h, txn_24h
    out = out.join(row_features(df))
    return df, out[FEATURES]


def row_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["foreign"] = (df["country"] != df["card_home_country"]).astype(int)
    out["hour"] = df["hour"]
    out["night"] = df["hour"].between(1, 5).astype(int)
    out["new_device"] = df["new_device"]
    out["account_age_days"] = df["account_age_days"]
    out["merchant_risk"] = df["merchant_category"].map(MERCHANT_RISK).fillna(1.0)
    out["merchant_code"] = df["merchant_category"].map({c: i for i, c in enumerate(CATS["merchant_category"])}).fillna(-1)
    out["channel_code"] = df["channel"].map({c: i for i, c in enumerate(CATS["channel"])}).fillna(-1)
    out["tiny_amount"] = (df["amount"] < 4).astype(int)
    return out


class FraudModel:
    """Supervised gradient boosting blended with an unsupervised isolation forest.

    The isolation forest never sees labels, so it still raises an eyebrow at
    behaviour that has no labelled precedent (the 'evolving attack' case).
    """

    def __init__(self, w_iso: float = 0.25, seed: int = 7):
        self.w_iso = w_iso
        self.gb = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.08, max_iter=250, l2_regularization=1.0,
            class_weight="balanced", random_state=seed)
        self.iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed)
        self._iso_lo = self._iso_hi = None

    def fit(self, X: pd.DataFrame, y):
        self.gb.fit(X, y)
        self.iso.fit(X)
        raw = -self.iso.score_samples(X)
        self._iso_lo, self._iso_hi = np.percentile(raw, [1, 99.9])
        return self

    def _iso_score(self, X):
        raw = -self.iso.score_samples(X)
        return np.clip((raw - self._iso_lo) / (self._iso_hi - self._iso_lo + 1e-9), 0, 1)

    def supervised(self, X):
        return self.gb.predict_proba(X)[:, 1]

    def score(self, X) -> np.ndarray:
        p = self.supervised(X)
        # max-blend: anything either detector is confident about gets surfaced
        return np.maximum(p, self.w_iso * self._iso_score(X) + (1 - self.w_iso) * p)

    def explain(self, X_row: pd.DataFrame, baseline: pd.Series, top: int = 5):
        """Occlusion attribution: how much does the score drop if a feature is neutralised?"""
        # log-odds scale: scores saturate near 1.0, so probability deltas would all look ~0
        logit = lambda p: float(np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))))
        base = float(self.score(X_row)[0])
        deltas = []
        for f in FEATURES:
            x2 = X_row.copy()
            x2[f] = baseline[f]
            deltas.append((f, logit(base) - logit(float(self.score(x2)[0]))))
        deltas.sort(key=lambda d: -abs(d[1]))
        return base, [{"feature": f, "label": LABELS[f], "impact": round(d, 4),
                       "value": float(X_row[f].iloc[0])} for f, d in deltas[:top] if abs(d) > 1e-4]
