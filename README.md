# Bitcoin Price Prediction — ML Model Comparison

A comparison of four machine learning approaches for predicting Bitcoin (BTC-USD) closing prices: **Linear Regression**, **Random Forest**, **XGBoost**, and **LSTM**. Originally developed as the practical part of my master's thesis (*"Predictive Cryptocurrency Model"*, FESB, University of Split, 2024).

## Overview

The goal was to evaluate how well different regression and deep learning approaches can predict the next-day closing price of Bitcoin using historical OHLCV (Open, High, Low, Close, Volume) data, and to compare their performance under the same data pipeline and evaluation metrics.

**Models implemented:**

| Model | Type |
|---|---|
| Linear Regression | Linear model |
| Random Forest Regressor | Ensemble (bagging) |
| XGBoost Regressor | Ensemble (boosting) |
| LSTM | Recurrent neural network |

## Dataset

- **Source:** [Yahoo! Finance](https://finance.yahoo.com/quote/BTC-USD/history/) via the `yfinance` library
- **Asset:** BTC-USD
- **Range:** 2014-09-17 to 2024-09-01 (~3,600+ daily samples) — the date range is **fixed** rather than fetched "to present," and `btc.csv` is committed to this repo, so results are reproducible regardless of when the notebooks are run
- **Raw features:** Open, High, Low, Close, Volume (Dividends and Stock Splits were dropped — no correlation with price)
- **Target:** Close price

## Pipeline

1. **Data collection** — fetched via `yfinance`, cached locally as `btc.csv`
2. **Cleaning** — dropped irrelevant columns, converted index to datetime, removed timezone info
3. **Feature engineering** — rolling averages over 2, 7, 60, and 365-day horizons; ratio features (price vs. rolling average) and trend features for each horizon, to capture short-, medium-, and long-term market cycles
4. **Scaling** — `MinMaxScaler` (0–1 range)
5. **Train/test split** — 80:20 for classical models, time-ordered (no shuffling); LSTM additionally uses a 70:20:10 train/test/validation split with a 30-day sliding window
6. **Hyperparameter tuning:**
   - Linear Regression, Random Forest, XGBoost → `RandomizedSearchCV` with `TimeSeriesSplit` (3 splits) to preserve temporal order, scored on negative MSE
   - LSTM → `keras_tuner` for hyperparameter search; data reshaped into 3D sequences (30-day lookback window)
7. **Evaluation metrics** — MAE, MSE, RMSE, R²

## Results

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 263.05 | 396.75 | 0.9995 |
| Random Forest | 712.65 | 1253.64 | 0.9950 |
| XGBoost | 976.87 | 1750.99 | 0.9902 |
| LSTM | 637.86 | 1083.09 | 0.9962 |

Linear Regression achieved the lowest error in this setup, though all four models reached R² > 0.99. These results should be taken with caution — the comparison was constrained by limited compute (Google Colab free tier), and hyperparameter search space, model architecture, and feature set could all be expanded for better generalization. Full methodology, mathematical background, and discussion are available in the [thesis PDF](docs/thesis.pdf).

## Known limitations

- **Tree-based models can't extrapolate.** Random Forest and XGBoost predict by averaging values learned from the training set, so they cannot output a price higher than the maximum seen during training. If this pipeline is re-run with a test period where Bitcoin trades above its training-set range (e.g. a new all-time high), RFR/XGBoost predictions will plateau near the highest training value instead of tracking the real price. Linear Regression and LSTM don't share this exact limitation, but all models still degrade on genuinely out-of-distribution price regimes. This is why the dataset range is fixed (see [Dataset](#dataset)) rather than fetched live — it keeps the comparison reproducible and consistent with the metrics reported above.
- **Single train/test split.** Results come from one chronological 80:20 split rather than multiple walk-forward folds, so reported metrics carry some variance not captured by a single run.
- **Limited hyperparameter search.** Constrained by free-tier Colab compute (`RandomizedSearchCV` with 200 iterations for RFR/XGBoost, 50 trials for LSTM), so the reported results are a reasonable baseline rather than a fully optimized model.

## Repository structure

```
bitcoin-price-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── btc.csv                    # fixed dataset (2014-09-17 to 2024-09-01)
│   ├── 01_linear_regression.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_lstm.ipynb
│   ├── 05_models_comparison.ipynb
│   └── *.pkl                      # cached predictions/results, used by 05_models_comparison.ipynb
└── docs/
    └── thesis.pdf
```

> **Note:** `btc.csv` and all `.pkl` result files are kept in the same folder as the notebooks (rather than a separate `data/` folder), because every file path in the code is relative to the notebook's own working directory.

## Tech stack

`Python 3.10` · `yfinance` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `tensorflow` / `keras` · `keras-tuner` · `matplotlib` · `seaborn`

Developed and run on Google Colab.

## Running locally

```bash
git clone https://github.com/<your-username>/bitcoin-price-prediction.git
cd bitcoin-price-prediction
pip install -r requirements.txt
jupyter notebook notebooks/
```

The `.pkl` result files are already included, so you can open `05_models_comparison.ipynb` directly to see the model comparison without re-running anything.

**To regenerate everything from scratch** (e.g. after changing a model), run the notebooks in this order, from inside the `notebooks/` folder, in a single uninterrupted session:

1. `01_linear_regression.ipynb`
2. `02_random_forest.ipynb`
3. `03_xgboost.ipynb`
4. `04_lstm.ipynb`
5. `05_models_comparison.ipynb`

`05_models_comparison.ipynb` loads the `.pkl` files saved by the other four, so it must run last.

## Author

Ivan Filipović — Faculty of Electrical Engineering, Mechanical Engineering and Naval Architecture (FESB), University of Split

Thesis advisor: prof. dr. sc. Maja Štula
