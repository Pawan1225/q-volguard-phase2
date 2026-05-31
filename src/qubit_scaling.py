import subprocess
import sys
from pathlib import Path

import pandas as pd


qubit_settings = [7, 9, 12]
symbol = "SPY"

all_results = []

for q in qubit_settings:
    print(f"\nRunning {q}-qubit QRC prototype on {symbol}...")

    subprocess.run(
        [
            sys.executable,
            "src/run_phase2_prototype.py",
            "--symbol",
            symbol,
            "--qrc-qubits",
            str(q),
        ],
        check=True,
    )

    result_path = Path("outputs") / "phase2_prototype_results.csv"
    df = pd.read_csv(result_path)

    qrc_row = df[df["model"].str.contains("QRC")].copy()
    qrc_row["qrc_qubits"] = q

    all_results.append(qrc_row)

combined = pd.concat(all_results, ignore_index=True)

output_path = Path("outputs") / "qubit_scaling.csv"
combined.to_csv(output_path, index=False)

print("\n=== Qubit Scaling Results ===")
print(combined)

print(f"\nSaved: {output_path}")