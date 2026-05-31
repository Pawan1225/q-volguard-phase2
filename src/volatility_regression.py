import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from run_phase2_prototype import esn_features, qrc_features


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def run_regression(symbol="SPY"):
    print(f"\nRunning volatility regression for {symbol}...")

    df = yf.download(
        symbol,
        start="2015-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False,
    )

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] for c in df.columns]

    df = df[["Close", "Volume"]].dropna()

    df["log_return"] = np.log(df["Close"]).diff()
    df["rolling_volatility"] = df["log_return"].rolling(20).std()
    df["momentum_5"] = df["Close"].pct_change(5)
    df["momentum_20"] = df["Close"].pct_change(20)
    df["volume_change"] = df["Volume"].pct_change()

    # Regression target: future realized volatility 5 trading days ahead
    df["future_volatility"] = df["rolling_volatility"].shift(-5)

    df = df.dropna()

    features = [
        "log_return",
        "rolling_volatility",
        "momentum_5",
        "momentum_20",
        "volume_change",
    ]

    X = df[features].values
    y = df["future_volatility"].values

    split = int(len(X) * 0.75)

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    E_train = esn_features(X_train)
    E_test = esn_features(X_test)

    Q_train = qrc_features(X_train, n_qubits=9, depth=4)
    Q_test = qrc_features(X_test, n_qubits=9, depth=4)

    results = []

    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results.append(
        {
            "symbol": symbol,
            "model": "Random Forest Regressor",
            "rmse": rmse(y_test, rf_pred),
        }
    )

    esn = Ridge(alpha=1.0)
    esn.fit(E_train, y_train)
    esn_pred = esn.predict(E_test)

    results.append(
        {
            "symbol": symbol,
            "model": "ESN + Ridge Readout",
            "rmse": rmse(y_test, esn_pred),
        }
    )

    qrc = Ridge(alpha=1.0)
    qrc.fit(Q_train, y_train)
    qrc_pred = qrc.predict(Q_test)

    results.append(
        {
            "symbol": symbol,
            "model": "QRC-inspired 9Q + Ridge Readout",
            "rmse": rmse(y_test, qrc_pred),
        }
    )

    return results


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    symbols = ["SPY", "QQQ", "DIA"]

    all_results = []

    for symbol in symbols:
        all_results.extend(run_regression(symbol))

    results = pd.DataFrame(all_results)
    results.to_csv(output_dir / "volatility_regression_results.csv", index=False)

    summary = (
        results.groupby("model")["rmse"]
        .mean()
        .reset_index()
        .sort_values("rmse")
    )

    summary.to_csv(output_dir / "volatility_regression_summary.csv", index=False)

    print("\n=== Volatility Regression Results ===")
    print(results)

    print("\n=== Average RMSE ===")
    print(summary)

    pivot = results.pivot(
        index="symbol",
        columns="model",
        values="rmse",
    )

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("RMSE")
    plt.title("Future Volatility Regression")
    plt.tight_layout()
    plt.savefig(output_dir / "volatility_regression_rmse.png", dpi=300)

    print("\nSaved:")
    print("outputs/volatility_regression_results.csv")
    print("outputs/volatility_regression_summary.csv")
    print("outputs/volatility_regression_rmse.png")


if __name__ == "__main__":
    main()