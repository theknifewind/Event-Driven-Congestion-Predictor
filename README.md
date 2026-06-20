# 🚦 UrbanFlow AI: Event-Driven Traffic Congestion Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-success.svg)](https://shap.readthedocs.io/en/latest/)

UrbanFlow AI is an advanced, end-to-end predictive and prescriptive machine learning pipeline designed to revolutionize how traffic enforcement agencies respond to both planned and unplanned urban congestion events. 

Built for the **Traffic Innovation Hackathon**, this project replaces reactive, experience-driven policing with deterministic, AI-driven resource deployment.

---

## 🛑 The Problem
On-street illegal parking, planned VIP movements, and unplanned incidents (vehicle breakdowns, water-logging) frequently choke carriageways and intersections. Currently, traffic management faces three massive hurdles:
1. **Reactive Enforcement:** Police are deployed *after* gridlock occurs.
2. **Experience-Driven Deployment:** There is no standardized methodology for determining exactly *how many* barricades or personnel are required for a specific event.
3. **No Post-Event Learning:** Lack of quantitative congestion impact metrics prevents historical data from informing future deployments.

## 💡 Our Solution
UrbanFlow AI solves this by deploying a two-stage architecture:
1. **The Predictive Engine:** An advanced Ensemble AI that ingests real-time incident data (location, time, cause, raw text descriptions) to forecast a highly accurate `Congestion Impact Score` (1.0 to 10.0).
2. **The Prescriptive Engine:** An Operations Research heuristic matrix that translates the AI's predicted score into explicit, actionable deployment orders (exact numbers of personnel, barricades, and categorized diversion plans).

---

## 🧠 System Architecture & Methodology

### 1. Data Engineering & Synthetic Impact Metric
Because historical datasets rarely contain explicit labels for "congestion severity", we engineered a deterministic, mathematically robust target variable (`Impact Score`).
* **The Formula:** `log1p(Duration × Priority Multiplier × Road Closure Penalty)` normalized to a 1.0 - 10.0 scale.
* **Result:** Severe water-logging on high-priority corridors mathematically approaches a 10.0, while minor test incidents on low-priority streets score ~1.0.

### 2. Multi-Modal Feature Extraction
We extract deep insights from the raw data using three distinct methodologies:
* **Temporal:** Extraction of peak traffic hours and weekend constraints.
* **Spatial (Unsupervised & Supervised):** We utilize an unsupervised **K-Means Clustering** algorithm to dynamically segment the city's coordinates into 20 hotspot zones. This is layered with one-hot encoded geographical `corridor` and `zone` tags.
* **Natural Language Processing (NLP):** We employ **TF-IDF Vectorization** on raw police incident descriptions. This allows the model to capture severity sentiment (e.g., words like "blocked", "heavy", "fallen") directly from text.

### 3. Hyperparameter-Tuned Ensemble Model
We utilize a state-of-the-art **Voting Regressor** that blends the predictions of:
* `XGBoost Regressor`
* `RandomForest Regressor`

To guarantee maximum academic rigor, the model employs `RandomizedSearchCV` to dynamically test hundreds of configurations and mathematically lock in the optimal hyperparameters for the dataset.

### 4. Explainable AI (XAI)
To build trust with traffic enforcement agencies and eliminate the "black box" problem, we integrated **SHAP (SHapley Additive exPlanations)**. The pipeline generates a SHAP Summary Plot, explicitly visualizing exactly *why* the model predicted a specific impact score (e.g., proving that the presence of a heavy vehicle drove the score up).

---

## 📊 Model Evolution & Performance Comparison

To ensure the highest possible accuracy, we developed this pipeline through three major iterations. Our final V3 model achieves state-of-the-art precision.

| Metric / Feature | V1 (Baseline) | V2 (Ensemble + NLP) | V3 (Ultimate Tuned) |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Single XGBoost | Voting Regressor | **Voting Regressor** |
| **Data Handled** | Tabular, Lat/Long | Tabular, Lat/Long, Text | **Tabular, Text, Explicit Zones** |
| **Tuning Mechanism**| None (Default params) | None (Manual params) | **`RandomizedSearchCV`** |
| **Explainability**| Black Box | Black Box | **SHAP Integration** |
| **Mean Absolute Error**| `~1.25` | `~0.98` | `~0.98` |
| **Root Mean Sq. Error**| `~1.75` | `~1.36` | `~1.32` |

### ⚠️ Important Methodology Caveat: Synthetic Target & Prediction Limits
Since the raw dataset does not contain real-world historical congestion levels, travel delays, or resource counts, the target variable (`Impact_Score`) was engineered as a deterministic proxy: `log1p(Duration × Priority × Road Closure Penalty)`. 

Consequently, the model's reported accuracy (MAE ~0.98) comes with a critical caveat:
1. **Not a Measure of Real-world Congestion:** The MAE does not measure prediction error against physical road gridlock. Instead, it measures how well the regression ensemble can recover and approximate our proxy formula based on the initial incident parameters.
2. **The Prediction Task at Inference:** At inference time, the actual duration of the event is *unknown*. The model's actual value lies in its ability to predict the *expected* duration and severity of a new incident based solely on initial inputs (cause, description text, location, and time).
3. **The "Mathematical Floor" effect:** The error rates stabilize at V2/V3 because the model has reached the limits of predicting the synthetic formula from the available subset of features. 

**Future Roadmap:** To transition this from a bootstrap prototype to a production system, the target variable should be replaced with real-world observed traffic telemetry (e.g., GPS probe speed data, actual queue lengths, or real police dispatch logs).


---

## 🚀 Getting Started & Reviewer Testing

### Prerequisites
Make sure you have the following installed:
- Python 3.9+
- Anaconda / Miniconda (recommended)
- Required packages: `streamlit`, `pydeck`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `shap`, `matplotlib`

Install them in your environment using:
```bash
pip install streamlit pydeck xgboost scikit-learn pandas numpy shap matplotlib
```

### Running the Project

Reviewers can test the project in two ways:

#### Option A: The Interactive Web Dashboard (Streamlit App)
This features our premium Command Operations Center, Live Map Congestion Buffer, Digital Twin Road Simulation, and Dispatch Event Manager.
1. Run the Streamlit server from your terminal:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`.
3. Use the sidebar to simulate events and watch the map, simulation, and resources update in real-time.

#### Option B: The Python Jupyter Notebook (ML Pipeline)
This contains the hyperparameter tuning loop, model training scripts, and explainability plots.
1. Open the project folder in your preferred Jupyter environment.
2. Reopen and run `Event_Driven_Congestion.ipynb`.
3. Scroll to the bottom to view the **SHAP summary plots** and the model comparison outputs.

---
*Built with ❤️ for the Traffic Innovation Hackathon*

