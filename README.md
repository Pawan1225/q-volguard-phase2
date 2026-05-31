# Q-VolGuard Phase 2

## Regime-Aware Quantum Reservoir Computing for Financial Volatility Forecasting

Q-VolGuard is a hybrid AI–Quantum forecasting framework designed to improve financial volatility prediction across changing market regimes. The system combines Quantum Reservoir Computing (QRC), classical machine learning baselines, regime-aware evaluation, and reproducible benchmarking to investigate the practical value of quantum-enhanced time-series modeling for financial decision support.

This repository contains the complete Phase 2 prototype, benchmark outputs, figures, documentation, and supplementary materials submitted to the qBraid × MITRE × JonesTrading Global Industry Challenge 2026.

---

## Project Overview

Financial markets exhibit highly nonlinear and regime-dependent behavior that can make volatility forecasting difficult for traditional models.

Q-VolGuard explores whether quantum reservoir dynamics can provide richer temporal representations than conventional machine learning approaches.

The framework:

- Uses Quantum Reservoir Computing (QRC) for sequence modeling.
- Evaluates performance across multiple volatility regimes.
- Benchmarks against classical machine learning methods.
- Measures predictive performance, robustness, and computational cost.
- Provides reproducible experimentation pipelines and evidence artifacts.

---

## Key Features

- Quantum Reservoir Computing (QRC)
- Regime-aware volatility forecasting
- Multi-market evaluation
- Classical baseline comparison
- Walk-forward validation
- Reproducible benchmarking pipeline
- Publication-quality figures and tables
- Competition-ready documentation

---

## Markets Evaluated

The framework was evaluated using publicly available ETF market data including:

- SPY
- QQQ
- DIA

Additional assets can be incorporated through the same pipeline.

---

## Repository Structure

```text
.
│   .gitignore
│   README.md
│
├───assets
│   ├───figures
│   └───tables
│
├───data
│   └───README.md
│
├───docs
│   ├───fujitsu
│   │       Q_VolGuard_Fujitsu_Report.pdf
│   │
│   └───supplementary
│
├───outputs
│   ├───figures
│   ├───metrics
│   └───results
│
├───src
│
└───tests
```

---

## Methodology

The Q-VolGuard workflow consists of five major stages:

### 1. Data Collection

Historical market data are collected using public financial data sources.

### 2. Feature Engineering

Features are generated from:

- Returns
- Realized volatility
- Rolling statistics
- Momentum indicators
- Regime indicators

### 3. Quantum Reservoir Construction

Input sequences are encoded into parameterized quantum circuits that act as dynamic reservoirs.

Quantum state measurements generate high-dimensional representations used for forecasting.

### 4. Prediction Layer

Reservoir outputs are passed to classical learning layers for volatility prediction.

### 5. Evaluation

Performance is measured using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Runtime

---

## Results Summary

The experimental evaluation demonstrated:

- Strong performance in high-volatility market regimes.
- Consistent forecasting quality across multiple ETFs.
- Competitive results against classical machine learning baselines.
- Robust behavior under varying market conditions.

Detailed benchmark results are available in the generated outputs and accompanying technical report.

---

## Reproducibility

To reproduce experiments:

```bash
git clone https://github.com/Pawan1225/q-volguard-phase2.git

cd q-volguard-phase2

python -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_full_pipeline.py
```

Generated outputs will be stored in:

```text
outputs/
```

---

## Supplementary Materials

The repository includes supplementary competition materials:

- Technical report
- Benchmark outputs
- Figures
- Evaluation artifacts
- Reproducibility assets

Related Fujitsu Quantum Simulator report:

```text
docs/Q_VolGuard_Fujitsu_Report.pdf
```

---

## Competition Submission

This repository accompanies the submission:

**Q-VolGuard: Regime-Aware Quantum Reservoir Computing for Financial Volatility Forecasting**

Submitted to:

**qBraid × MITRE × JonesTrading Global Industry Challenge 2026**

---

## Future Work

Planned extensions include:

- Larger quantum reservoirs
- Additional asset classes
- Multi-horizon forecasting
- Portfolio-level risk forecasting
- Hybrid HPC–Quantum execution
- Advanced uncertainty quantification

---



---

## License

This repository is released for research and educational purposes.

Please contact the authors regarding commercial use.

---

## Citation

If you use this work, please cite:

```text
Q-VolGuard: Regime-Aware Quantum Reservoir Computing for Financial Volatility Forecasting.

qBraid × MITRE × JonesTrading Global Industry Challenge 2026.
```