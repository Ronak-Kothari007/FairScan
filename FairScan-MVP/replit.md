# FairScan — AI Bias Detection Dashboard

## Overview

FairScan is a Streamlit dashboard that audits tabular ML models for fairness.
It trains a logistic regression classifier on a user-supplied or sample dataset,
computes fairness metrics with `fairlearn`, visualizes group-level approval rates
before and after mitigation (`ThresholdOptimizer` with demographic-parity
constraint), and exports a one-click PDF audit report (`fpdf2`).

## Stack

- **Language**: Python 3.11
- **App framework**: Streamlit (dark theme)
- **ML**: scikit-learn (LogisticRegression, ColumnTransformer, OneHotEncoder)
- **Fairness**: fairlearn (metrics + ThresholdOptimizer post-processing)
- **Charts**: Plotly (grouped bar chart, dark template)
- **PDF**: fpdf2

## Project Layout

- `app.py` — single-file Streamlit application (sidebar controls + main report)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — server + dark theme configuration

## Running & Deployment

This is a pnpm monorepo, so the Streamlit app is registered as a service inside
the `api-server` artifact (`artifacts/api-server/.replit-artifact/artifact.toml`).
That artifact exposes two services:

- `FairScan` — Streamlit on port 5000, served at `/` (the user-facing app)
- `API Server` — Node Express on port 8080, served at `/api` (unused by FairScan
  itself, kept for future expansion)

Both `[services.development]` and `[services.production]` blocks are configured,
which is what the Replit publisher reads when the user clicks Publish. Note that
the production run command uses `../../app.py` because the artifact's working
directory is `artifacts/api-server/`.

Local dev command (run by the artifact-managed workflow):

```bash
streamlit run ../../app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Default Dataset

UCI Adult Income is fetched on first run and cached. If the network fetch fails,
the app falls back to a synthetic biased dataset so the demo always works.

## Bias Thresholds

- `> 0.10` → **High Risk** (red)
- `0.05 – 0.10` → **Needs Review** (yellow)
- `< 0.05` → **Fair** (green)
