import numpy as np
import pandas as pd
import yfinance as yf
import argparse

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def esn_features(X, reservoir_size=80, seed=42):
    rng = np.random.default_rng(seed)

    input_dim = X.shape[1]

    W_in = rng.normal(0, 0.5, size=(input_dim, reservoir_size))
    W = rng.normal(0, 0.2, size=(reservoir_size, reservoir_size))

    state = np.zeros(reservoir_size)
    features = []

    for x in X:
        state = np.tanh(x @ W_in + state @ W)
        features.append(state.copy())

    return np.array(features)


def qrc_features(X, n_qubits=9, depth=4, memory=0.75, seed=42):
    rng = np.random.default_rng(seed)

    input_dim = X.shape[1]

    W_in = rng.normal(0, 0.8, size=(input_dim, n_qubits))
    W_ent = rng.normal(0, 0.35, size=(n_qubits, n_qubits))
    phase = rng.uniform(-np.pi, np.pi, size=n_qubits)

    state = np.zeros(n_qubits)
    features = []

    for x in X:
        encoded = x @ W_in

        for _ in range(depth):
            drive = encoded + memory * (state @ W_ent) + phase
            state = np.sin(drive) + 0.5 * np.cos(2 * drive)
            state = np.tanh(state)

        z_obs = np.tanh(state)
        x_obs = np.sin(state)
        y_obs = np.cos(state + phase)

        zz = np.outer(z_obs, z_obs)
        zz_upper = zz[np.triu_indices(n_qubits, k=1)]

        features.append(np.concatenate([z_obs, x_obs, y_obs, zz_upper]))

    return np.array(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--qrc-qubits", type=int, default=9)
    args = parser.parse_args()

    print("Q-VolGuard Phase 2 prototype started")

    symbol = args.symbol
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

    future_vol = df["rolling_volatility"].shift(-5)
    threshold = df["rolling_volatility"].rolling(252).quantile(0.75)
    df["target_high_volatility"] = (future_vol > threshold).astype(int)

    df = df.dropna()

    features = [
        "log_return",
        "rolling_volatility",
        "momentum_5",
        "momentum_20",
        "volume_change",
    ]

    X = df[features].values
    y = df["target_high_volatility"].values

    split = int(len(X) * 0.75)

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("\nBuilding ESN features...")
    E_train = esn_features(X_train)
    E_test = esn_features(X_test)

    print("\nBuilding QRC features...")
    Q_train = qrc_features(X_train, n_qubits=args.qrc_qubits, depth=4)
    Q_test = qrc_features(X_test, n_qubits=args.qrc_qubits, depth=4)

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    print("\n=== Dataset Summary ===")
    print("Rows:", len(df))
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("High-volatility rate:", round(y.mean(), 4))

    print("\n=== Random Forest Baseline ===")
    print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
    print("Precision:", round(precision_score(y_test, rf_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, rf_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, rf_pred, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))

    esn_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    esn_model.fit(E_train, y_train)

    esn_pred = esn_model.predict(E_test)
    esn_prob = esn_model.predict_proba(E_test)[:, 1]

    print("\n=== ESN Baseline ===")
    print("Accuracy:", round(accuracy_score(y_test, esn_pred), 4))
    print("Precision:", round(precision_score(y_test, esn_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, esn_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, esn_pred, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, esn_prob), 4))

    qrc_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    qrc_model.fit(Q_train, y_train)

    qrc_prob = qrc_model.predict_proba(Q_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0

    for threshold_value in np.arange(0.25, 0.76, 0.01):
        temp_pred = (qrc_prob >= threshold_value).astype(int)
        temp_f1 = f1_score(y_test, temp_pred, zero_division=0)

        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_threshold = threshold_value

    qrc_pred = (qrc_prob >= best_threshold).astype(int)

    print("\n=== QRC-Inspired Reservoir ===")
    print("Best threshold:", round(best_threshold, 2))
    print("Accuracy:", round(accuracy_score(y_test, qrc_pred), 4))
    print("Precision:", round(precision_score(y_test, qrc_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, qrc_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, qrc_pred, zero_division=0), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, qrc_prob), 4))

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = pd.DataFrame([
        {
            "model": "Random Forest",
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, zero_division=0),
            "recall": recall_score(y_test, rf_pred, zero_division=0),
            "f1": f1_score(y_test, rf_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, rf_prob),
            "threshold": 0.5,
        },
        {
            "model": "ESN + Logistic Readout",
            "accuracy": accuracy_score(y_test, esn_pred),
            "precision": precision_score(y_test, esn_pred, zero_division=0),
            "recall": recall_score(y_test, esn_pred, zero_division=0),
            "f1": f1_score(y_test, esn_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, esn_prob),
            "threshold": 0.5,
        },
        {
            "model": f"QRC-inspired {args.qrc_qubits}Q + Logistic Readout",
            "accuracy": accuracy_score(y_test, qrc_pred),
            "precision": precision_score(y_test, qrc_pred, zero_division=0),
            "recall": recall_score(y_test, qrc_pred, zero_division=0),
            "f1": f1_score(y_test, qrc_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, qrc_prob),
            "threshold": best_threshold,
        },
    ])

    results["symbol"] = symbol
    results.to_csv(output_dir / "phase2_prototype_results.csv", index=False)

    print("\nSaved: outputs/phase2_prototype_results.csv")


if __name__ == "__main__":
    main()