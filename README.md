
<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue.svg"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://github.com/emilio027/Time_Series_Stock_Analysis/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/emilio027/Time_Series_Stock_Analysis/ci.yml?label=CI"></a>
  <a href="https://github.com/emilio027/Time_Series_Stock_Analysis/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/emilio027/Time_Series_Stock_Analysis"></a>
  <a href="#"><img alt="Code style" src="https://img.shields.io/badge/style-ruff-informational"></a>
</p>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilio027/Time_Series_Stock_Analysis/blob/main/notebooks/quickstart.ipynb)


![Preview](docs/img/preview.png)

    # Time Series Stock Analysis — Forecasts, Drawdowns & Regimes

    ## Executive Summary
    Reproducible pipeline for equity series: ingestion (OHLCV), feature engineering, baseline & SARIMAX/Prophet
forecasts, uncertainty visualization, and drawdown analysis. Emphasis on coverage, residual diagnostics, and
where models fail.

    **ATS Keywords:** Python, SQL, Power BI, Tableau, Pandas, NumPy, scikit-learn, ETL, data pipeline, automation, business intelligence, KPI dashboard, predictive modeling, time series forecasting, feature engineering, stakeholder management, AWS, GitHub Actions, Streamlit, Prophet, SARIMAX, SHAP, risk analytics, calibration, cross-validation, A/B testing

    ## Skills & Tools
    - Python
- Pandas
- statsmodels
- Prophet
- Feature engineering
- Walk-forward CV
- Uncertainty quantification

    ## Deliverables
    - Scripts to fetch and prepare OHLCV data (yfinance or CSV)
- Baseline forecasts (naive, SMA) plus SARIMAX/Prophet variants
- Evaluation with coverage tests, residuals, ACF/PACF, and backtests
- Visual pack: forecasts with CIs, rolling Sharpe, drawdowns, regime hints

    ## Key Metrics / Evaluation
    - MAE/MAPE
- Coverage probability
- Backtest returns
- Max Drawdown

    ## How to Run
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    make data
    make report
    ```
    *Law Firm demo:* `streamlit run app.py`

    ## Impact Highlights (from my work history)
    - Saved $3M by automating workflows and migrating Excel processes to SAP HANA at NRG
- Resolved data issues saving $500k annually at CenterPoint Energy
- Improved stakeholder transparency by 15% via SQL + Power BI/Tableau dashboards at Robin Hood
- Scaled an AI automation agency from $750 to $28k weekly revenue as Founder/CEO

    ## Repo Structure
    ```
    src/  notebooks/  data/{raw,processed}  models/  scripts/  tests/  docs/img/  reports/
    ```
