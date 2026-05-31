import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/qubit_scaling.csv")

plt.figure(figsize=(7, 4))
plt.plot(df["qrc_qubits"], df["f1"], marker="o")
plt.xlabel("QRC Reservoir Size (Qubits)")
plt.ylabel("F1 Score")
plt.title("Small-Scale QRC Prototype: Qubit Scaling")
plt.tight_layout()
plt.savefig("outputs/qubit_scaling.png", dpi=300)

print("Saved outputs/qubit_scaling.png")