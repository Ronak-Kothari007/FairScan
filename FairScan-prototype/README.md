# FairScan

FairScan is a Streamlit prototype for auditing binary classification datasets for group fairness risks.

## Features

- CSV upload with sidebar controls for sensitive attribute, target column, and positive outcome.
- Built-in Adult Income-style demo sample so the app runs immediately without external APIs.
- Logistic regression training with `pandas`, `scikit-learn`, and Fairlearn metrics.
- Metric cards for demographic parity difference, equalized odds difference, and group calibration difference.
- Plotly grouped bar chart comparing positive outcome rates before and after Fairlearn `ThresholdOptimizer` mitigation.
- One-click downloadable PDF report with verdict and recommendations.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
