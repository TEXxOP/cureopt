# 🏗️ CureOpt AI

**AI-Powered Cycle Time Optimization for Precast Yards**

L&T CreaTech 2025 · Problem Statement 1

---

## 🎯 What It Does

CureOpt AI is a machine-learning-powered decision engine that:
- **Predicts** concrete strength development using an XGBoost + Random Forest ensemble (R² = 0.98)
- **Optimizes** curing strategy using a Genetic Algorithm (DEAP) to minimize cycle time × cost
- **Recommends** the optimal de-moulding time with full cost visibility and SHAP explainability
- **Adapts** automatically for 3 Indian climatic zones (Chennai, Delhi, Mumbai)

## 📊 Key Results

| Metric | Industry Standard | CureOpt AI |
|--------|-------------------|------------|
| Cycle time | 24–36 hours | **3.5 hours** (−85.4%) |
| Mold turns/day | 1.3× | **3.69×** |
| Cost per cycle | Baseline | **0.457× baseline** (−54%) |
| Strength prediction R² | No model | **0.98** |

## 🏗️ Architecture

```
Input (9 params) → ML Predictor (XGBoost+RF) → Cost Model → GA Optimizer → Dashboard
```

**5 Layers:**
1. **Data Inputs** — Mix design, curing method, climate, required MPa
2. **Strength Prediction** — Ensemble ML model trained on 5,000-row synthetic dataset
3. **Cost Model** — Parametric: material + energy + labor + mold occupancy (region-aware)
4. **GA Optimizer** — DEAP: 100 population × 50 generations, early stopping
5. **Streamlit Dashboard** — Interactive UI with Plotly charts + PDF export

## 🚀 Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Generate dataset
python data/generate_dataset.py

# 3. Train models (~15-20 min)
python models/train_model.py

# 4. Launch dashboard
python -m streamlit run app.py
```

Or with Docker:
```bash
docker build -t cureopt-ai .
docker run -p 8501:8501 cureopt-ai
```

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** scikit-learn, XGBoost, Optuna, SHAP
- **Optimization:** DEAP (Genetic Algorithm)
- **Dashboard:** Streamlit, Plotly
- **Export:** ReportLab (PDF)
- **Deployment:** Docker

## 📁 Project Structure

```
├── app.py                      # Streamlit dashboard
├── config.py                   # Central configuration
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container setup
├── data/
│   └── generate_dataset.py     # Synthetic data generation
├── models/
│   ├── train_model.py          # ML training pipeline
│   └── predict.py              # Inference + SHAP
├── optimizer/
│   ├── cost_model.py           # Parametric cost function
│   └── ga_optimizer.py         # DEAP GA + grid fallback
└── utils/
    └── pdf_report.py           # PDF report generation
```

## 👥 Team CureOpt AI

L&T CreaTech 2025

---

*CureOpt AI — Deployable in 90 days. ROI in Year 1.*
