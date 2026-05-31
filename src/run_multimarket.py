import subprocess
import sys
from pathlib import Path

import pandas as pd


symbols = ["SPY", "QQQ", "DIA"]

all_results = []

for symbol in symbols:
    print(f"\nRunning prototype for {symbol}...")

    subprocess.run(
        [
            sys.executable,
            "src/run_phase2_prototype.py",
            "--symbol",
            symbol,
        ],
        check=True,
    )

    result_path = Path("outputs") / "phase2_prototype_results.csv"
    df = pd.read_csv(result_path)

    df["symbol"] = symbol
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

output_path = Path("outputs") / "phase2_multimarket_results.csv"
combined.to_csv(output_path, index=False)

summary = (
    combined.groupby("model")[["f1", "roc_auc"]]
    .mean()
    .reset_index()
)

summary = summary.sort_values("f1", ascending=False)

summary_path = Path("outputs") / "phase2_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n=== Combined Multi-Market Results ===")
print(combined)

print(f"\nSaved: {output_path}")

print("\n=== Average Results ===")
print(summary)

print(f"\nSaved: {summary_path}")