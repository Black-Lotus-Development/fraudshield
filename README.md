# FraudShield — DataNova 2026, DN-FN-01 Real-Time Payment Fraud Detection

Real-time fraud scoring with low false positives, drift adaptation and cost-aware decisions.

## Run
```bash
pip install -r requirements.txt
python train.py     # generates data, trains, validates, writes artifacts/
python app.py       # dashboard at http://localhost:5050
```

## Approach
- **Data:** synthetic 80k-transaction / 60-day portfolio, 1.6% fraud, four typologies (account takeover, card testing, night cash-out, and a *new* mobile-travel pattern that only appears in the last quarter). Synthetic by necessity — no real payment data was available, so absolute scores overstate real-world separability.
- **Features (causal only):** per-card velocity (1h/24h), amount vs. card's running average, geo mismatch, device, merchant risk, hour, account age.
- **Model:** class-balanced HistGradientBoosting blended (max) with an unsupervised IsolationForest for novel behaviour.
- **Validation:** strictly time-ordered split — train 0–65%, validation 65–75% (threshold chosen to minimise *missed fraud $ + $5/review*), test A 75–87.5%, test B 87.5–100%.
- **Drift:** static model (trained on early data) vs. adaptive model (retrained once test-A labels arrive). Static recall on the new pattern ≈ 0–1%; adaptive ≈ 100%.
- **Explainability:** per-transaction occlusion attribution on the log-odds scale; permutation importance globally.
- **Responsible AI:** FPR/recall by channel and home country, no protected attributes, borderline scores routed to human review.

## Caveats
Single synthetic dataset, one random seed; the drift scenario is constructed, so the adaptive model's near-perfect recall on it should be read as a demonstration of the retraining loop, not a benchmark.
