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
1. **The Predictive Engine:** An advanced Ensemble AI that ingests real-time incident data (location, time, cause, raw text descriptions) to forecast a `Congestion Impact Score` (1.0 to 10.0).
2. **The Prescriptive Engine:** An Operations Research heuristic matrix that translates the AI's predicted score into explicit, actionable deployment orders (exact numbers of personnel, barricades, and categorized diversion plans)## 🧠 System Architecture & Methodology

### 1. Data Engineering & Preprocessing Refactoring
To ensure high-fidelity predictive modeling without target leakage or bias, the data preprocessing pipelines in [app.py](file:///c:/Users/sriji/Projects/Event-Driven%20Congestion/app.py) and [Event_Driven_Congestion.ipynb](file:///c:/Users/sriji/Projects/Event-Driven%20Congestion/Event_Driven_Congestion.ipynb) were refactored with the following operations:
* **Planned Event Schema Resolution (Coalesced Timestamps):** Planned events use `end_datetime` while unplanned events use `closed_datetime`. By coalescing these fields into `resolved_time = closed_datetime.fillna(end_datetime)`, we preserved planned events (rallies, festivals, construction) in our training data, preventing 93% of planned events from being dropped. This increased usable planned events in the training data from 33 to 297.
* **Data Filtration Drop Statistics:** Around 63% of raw incidents are active and lack resolution times. We drop these during training because duration cannot be computed, leaving a clean training subset containing 36.6% of the raw data (2,994 records).
* **Target Leakage Elimination:** Previously, training the model directly on a synthetic target variable (`Impact_Score`) that was calculated using `priority` and `closure_multiplier` led to target leakage since these were also model features. The V3 model has been retrained to predict the physical event clearing time (`log1p(duration_mins)`) directly. The final `Impact_Score` is computed downstream using deterministic policy multipliers.

### 2. Multi-Modal Feature Engineering
We extract features from three distinct facets of each incident:
* **Temporal Signals:** Hour of the day, peak hours (8-11 AM, 5-8 PM), and weekend indicators.
* **Spatial Hotspots:** Unsupervised **K-Means Clustering** (20 clusters) to partition the city geographically, combined with one-hot encoded geographical `zone` and `corridor` tags.
* **Natural Language Processing (NLP):** **TF-IDF Vectorization** (top 50 features) on raw incident descriptions to extract qualitative severity indicators (e.g., "heavy", "blocked", "waterlogged").
* **Real-time Spatial-Temporal Load (`nearby_active_events`):** A dynamic concurrent load feature calculating the count of active, unresolved incidents within a 5 km radius at the event's start time.

### 3. Hyperparameter-Tuned Ensemble Model
We employ a **Voting Regressor** combining:
* `XGBoost Regressor` (tuned using `RandomizedSearchCV` on estimator count, depth, and learning rate).
* `RandomForest Regressor` (configured with `n_estimators=100`, `max_depth=10`).

### 4. Explainable AI (XAI)
To provide operational transparency, we use **SHAP (SHapley Additive exPlanations)**. Because VotingRegressor does not support TreeExplainer directly, we explain the RandomForest component (`model.estimators_[1]`) as a proxy for the ensemble's feature importances, showing how peak hours, text descriptors, and geographic features impact predictions.

---

## 📊 Model Evolution & Performance Comparison

To ensure operational transparency and robustness, we developed this pipeline through three major iterations. The final V3 model eliminates target leakage and predicts duration directly.

| Metric / Feature | V1 (Baseline) | V2 (Ensemble + NLP) | V3 (Leakage-Free Tuned) |
| :--- | :--- | :--- | :--- |
| **Model Target** | `Impact_Score` (Direct) | `Impact_Score` (Direct) | **`log1p(duration_mins)` (Downstream Scaling)** |
| **Target Leakage** | Yes (High leakage) | Yes (High leakage) | **None (Fully Resolved)** |
| **Model Architecture** | Single XGBoost | Voting Regressor | **Voting Regressor (XGBoost + RF)** |
| **Data Handled** | Tabular, Lat/Long | Tabular, Lat/Long, Text | **Tabular, Text, Zones, Concurrent Load** |
| **Tuning Mechanism**| None (Default params) | None (Manual params) | **`RandomizedSearchCV`** |
| **Explainability**| None | None | **SHAP Explainer (RandomForest Proxy)** |
| **Naive Baseline MAE**| `~1.23` (Mean Pred) | `~1.23` (Mean Pred) | **`~1.23` (Mean Pred)** |
| **Downstream Impact MAE**| `~1.25` | `~0.98` | **`1.007`** |
| **Improvement vs Baseline**| `-1.6%` | `+20.3%` (Leaked) | **~18%** (Defensible) |
| **Downstream Impact RMSE**| `~1.75` | `~1.36` | **`1.399`** |

### ⚠️ Important Methodology Caveat & Target Leakage Prevention
Since the raw dataset does not contain real-world historical congestion levels, travel delays, or resource counts, the final `Impact_Score` is engineered as a policy-driven proxy: `log1p(Duration × Priority × Road Closure Penalty)`. 

In previous versions, training the model directly on this formula resulted in target leakage, as `priority` and `closure_multiplier` were also model inputs. We have fully resolved this:
1. **Leakage-Free Duration Prediction:** The ML models in V3 are trained to predict the physical event clearing time (`log1p(duration_mins)`) using initial parameters (incident cause, text, time, and coordinates).
2. **Downstream Scaling Policy:** Once the duration is predicted, the `Impact_Score` is computed downstream using deterministic policy multipliers (priority and road closure constraints).
3. **Evaluation Caveat:** The final impact score MAE of `1.007` measures how closely the ML model can reconstruct the downstream proxy using predicted durations. Since actual duration is unknown when an incident is first reported, the ML model serves to forecast the expected resolution time.
4. **Data Filtration Statistics:** About 63% of the raw data (specifically active events that lack a closed or end resolution timestamp) is filtered out because their durations cannot be computed. This leaves 36.6% (2,994 records) of the dataset available for training.

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
2. Reopen and run [Event_Driven_Congestion.ipynb](file:///c:/Users/sriji/Projects/Event-Driven%20Congestion/Event_Driven_Congestion.ipynb).
3. Scroll to the bottom to view the **SHAP summary plots** and the model comparison outputs.


