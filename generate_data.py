"""Synthetic payment-transaction generator with realistic fraud typologies.

Fraud is ~1.6% of volume and is injected as four distinct patterns so that a
model has to generalise rather than memorise one rule:
  * account takeover   - new device + foreign IP + high amount
  * card testing       - bursts of tiny amounts on one card
  * night-time cash-out- large amount, risky merchant, odd hours
  * drift pattern      - a new pattern only present in the last 25% of time
                         (simulates evolving attacks; tests adaptivity)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 80_000
MERCHANTS = ["grocery", "fuel", "restaurant", "travel", "electronics",
             "gift_cards", "crypto", "online_retail", "pharmacy", "utilities"]
RISKY = {"gift_cards": 3.0, "crypto": 3.5, "electronics": 1.8, "travel": 1.4}
COUNTRIES = ["US", "US", "US", "US", "GB", "DE", "IN", "BR", "NG", "RU", "CN"]


def generate(n: int = N) -> pd.DataFrame:
    t = np.sort(RNG.uniform(0, 60 * 24 * 3600, n))  # 60 days, seconds
    hour = ((t // 3600) % 24).astype(int)
    n_cards = 6000
    card = RNG.integers(0, n_cards, n)
    home = RNG.choice(COUNTRIES[:5], n_cards)
    card_home = home[card]
    avg_spend = RNG.lognormal(3.4, 0.5, n_cards)[card]

    merchant = RNG.choice(MERCHANTS, n, p=[.22, .12, .16, .05, .06, .02, .01, .22, .06, .08])
    amount = RNG.lognormal(np.log(avg_spend), 0.8)
    country = np.where(RNG.random(n) < 0.94, card_home, RNG.choice(COUNTRIES, n))
    new_device = (RNG.random(n) < 0.05).astype(int)
    channel = RNG.choice(["pos", "online", "mobile", "atm"], n, p=[.40, .32, .24, .04])
    acct_age_days = RNG.integers(5, 3000, n)
    fraud = np.zeros(n, dtype=int)
    pattern = np.array(["legit"] * n, dtype=object)

    def mark(idx, name):
        fraud[idx] = 1
        pattern[idx] = name

    late = np.arange(n) > int(n * 0.75)

    # 1) account takeover
    idx = RNG.choice(np.arange(n), int(n * 0.006), replace=False)
    new_device[idx] = 1
    country[idx] = RNG.choice(["NG", "RU", "CN", "BR"], len(idx))
    amount[idx] = amount[idx] * RNG.uniform(3, 9, len(idx))
    channel[idx] = "online"
    mark(idx, "account_takeover")

    # 2) card testing: tiny amounts, clustered per card
    idx = RNG.choice(np.arange(n), int(n * 0.004), replace=False)
    amount[idx] = RNG.uniform(0.5, 3.0, len(idx))
    channel[idx] = "online"
    merchant[idx] = RNG.choice(["online_retail", "gift_cards"], len(idx))
    card[idx] = RNG.integers(0, 120, len(idx))  # reuse few cards => velocity
    mark(idx, "card_testing")

    # 3) night cash-out
    idx = RNG.choice(np.arange(n), int(n * 0.004), replace=False)
    hour[idx] = RNG.choice([1, 2, 3, 4], len(idx))
    amount[idx] = amount[idx] * RNG.uniform(4, 12, len(idx))
    merchant[idx] = RNG.choice(["crypto", "gift_cards", "electronics"], len(idx))
    mark(idx, "night_cashout")

    # 4) drift: unseen pattern only in the last quarter of the timeline
    idx = RNG.choice(np.where(late)[0], int(n * 0.006), replace=False)
    channel[idx] = "mobile"
    merchant[idx] = "travel"
    country[idx] = RNG.choice(["BR", "NG"], len(idx))
    amount[idx] = amount[idx] * RNG.uniform(2.5, 6, len(idx))
    acct_age_days[idx] = RNG.integers(5, 40, len(idx))
    mark(idx, "drift_mobile_travel")

    # Light label noise on legit side: some legit big/foreign txns look suspicious
    idx = RNG.choice(np.where(fraud == 0)[0], int(n * 0.01), replace=False)
    amount[idx] = amount[idx] * RNG.uniform(3, 7, len(idx))
    country[idx] = RNG.choice(COUNTRIES, len(idx))

    df = pd.DataFrame({
        "txn_id": [f"T{i:06d}" for i in range(n)],
        "timestamp": t.astype(int),
        "card_id": card,
        "amount": amount.round(2),
        "merchant_category": merchant,
        "channel": channel,
        "country": country,
        "card_home_country": card_home,
        "hour": hour,
        "new_device": new_device,
        "account_age_days": acct_age_days,
        "is_fraud": fraud,
        "pattern": pattern,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(df.is_fraud.mean().round(4), len(df))
    print(df[df.is_fraud == 1].pattern.value_counts())

