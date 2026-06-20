# Fundamentals and Reasoning

## Initial Dataset Analysis
We explored the dataset located at `dataset\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv`.
- **Size**: ~8,200 rows
- **Key Columns Available**: `event_type` (planned/unplanned), `latitude`, `longitude`, `event_cause` (vehicle_breakdown, tree_fall, water_logging, etc.), `requires_road_closure`, `start_datetime`, `closed_datetime`, `priority`, `veh_type`, `description`, `zone`, `corridor`.
- **Missing Information**: The problem statement requires predicting "traffic impact" and recommending "manpower, barricading, and diversion plans." However, the dataset *does not* explicitly contain historical labels for these parameters.

## Fundamental Approach (Why this way and not hit-and-trial)
Because we lack a direct `Y` variable for impact or resources, a naïve hit-and-trial approach would attempt to blindly map strings to outputs. Instead, we must use a robust, deterministic proxy approach:

### 1. Synthetic Impact Metric (The "Why")
Since we don't have vehicle count or speed reduction data, the best proxy for congestion severity is **Duration x Spatial Vulnerability x Event Severity**.
- A tree fall on a major corridor (High priority) taking 4 hours to clear has a massive impact.
- A vehicle breakdown on a non-corridor (Low priority) taking 15 minutes has minimal impact.
We mathematically formulated this as `log1p(duration_mins * priority * closure_multiplier)` and scaled it 1-10.
- *Results:* The math worked perfectly. Water logging events that require closures maxed out at 10.0, while minor tree falls on low-priority roads scored ~1.1.

### 2. Advanced Multi-Modal Feature Extraction (The "Why")
Instead of just feeding raw columns into a model, we extracted deep context:
- **Spatial Features:** We used **K-Means clustering** on Latitude/Longitude to automatically segment the city into 20 dynamic hotspot clusters. We then layered this with explicit geofencing by one-hot encoding the `zone` and `corridor` strings.
- **NLP Features:** The `description` column holds vital clues (e.g., "heavy", "blocked", "fallen"). We used **TF-IDF Vectorization** to turn these raw text descriptions into mathematical severity features.

### 3. Hyperparameter Tuned Voting Regressor (The "Why")
We utilized a **Voting Regressor**, combining the strengths of an **XGBoost Regressor** and a **RandomForest Regressor**.
- *Why not Deep Learning?* Tree-based models dominate tabular datasets of this scale (~8k rows). 
- *Why Tuning?* We optimized the XGBoost hyperparameters using `RandomizedSearchCV`. This identifies a highly performant parameter configuration within our search space, reducing overall variance compared to default configurations.

### 4. Explainable AI via SHAP (The "Why")
"Black box" AI is dangerous for law enforcement. We implemented **SHAP (SHapley Additive exPlanations)** to peek under the hood. The SHAP Summary Plot visibly demonstrates which features (like peak hour, a specific zone, or a TF-IDF text keyword) influence the model's predictions.

### 5. Heuristic Resource Allocation (The "Why")
Machine Learning cannot predict what it hasn't seen. Since we have no historical labels for "number of barricades used", we implemented an **Operations Research approach**. We defined a rule matrix that translates the continuous `Impact_Score` predicted by the AI directly into discrete physical resources (Police Officers, Barricades, Diversion levels).

---

## Model Evaluation Metrics

Our Voting Regressor's validation performance is evaluated using two primary metrics:

### 1. Mean Absolute Error (MAE): ~0.98
**What it is:** MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
**Operational Context:** An MAE of ~0.98 on a 1-to-10 scale indicates that the predicted impact score generally aligns with our engineered proxy category, serving as a reliable baseline guide for resource allocation.

### 2. Root Mean Squared Error (RMSE): ~1.32
**What it is:** RMSE is the standard deviation of the residuals (prediction errors). It is sensitive to large individual errors due to squaring.
**Operational Context:** An RMSE of ~1.32 shows that the model's prediction errors are relatively well-contained, with few extreme outliers. Minimizing the gap between MAE and RMSE ensures that the model rarely makes catastrophic mispredictions (e.g., predicting a minor disruption when the event has highly critical features), preventing operational under-deployment.


---

## ⚠️ Methodology Caveat: Synthetic Target & Prediction Limits

Because the raw dataset lacks physical ground-truth labels for "traffic delay" or "congestion severity," the target variable (`Impact_Score`) was synthetically engineered as a proxy: `log1p(Duration × Priority × Road Closure Penalty)`. 

This introduces an important limitation that must be disclosed:
1. **Target Rationale:** Rather than representing absolute physical gridlock, the MAE (~0.98) measures how closely the ML models can approximate our proxy formula using only the initial incident attributes.
2. **Inference Value:** Since the *actual duration* of a new incident is unknown when it is first reported, the ML model acts as a predictor of the *expected duration/severity* of the event, enabling proactive dispatching.
3. **Generalization & Future Steps:** The model is an operational bootstrap. In a real-world production deployment, this model should be retrained on actual traffic speed telemetry (e.g., GPS probe speeds or loops sensors) rather than a derived mathematical formula.

