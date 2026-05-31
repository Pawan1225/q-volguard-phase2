import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("outputs/phase2_multimarket_results.csv")

pivot = df.pivot(
    index="symbol",
    columns="model",
    values="f1",
)

ax = pivot.plot(
    kind="bar",
    figsize=(8, 5),
)

plt.ylabel("F1 Score")
plt.title("Multi-Market Volatility Regime Detection")
plt.tight_layout()

plt.savefig(
    "outputs/multimarket_f1.png",
    dpi=300,
)

print("Saved outputs/multimarket_f1.png")