# Fundamentals and Reasoning

## Initial Dataset Analysis
We explored the dataset located at [Astram event data](file:///c:/Users/sriji/Projects/Event-Driven%20Congestion/dataset/Astram%20event%20data_anonymized%20-%20Astram%20event%20data_anonymizedb40ac87.csv) (relative path: `dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv`).
- **Size**: ~8,200 raw rows.
- **Key Columns Available**: `event_type` (planned/unplanned), `latitude`, `longitude`, `event_cause` (vehicle_breakdown, tree_fall, water_logging, etc.), `requires_road_closure`, `start_datetime`, `closed_datetime`, `end_datetime`, `priority`, `veh_type`, `description`, `zone`, `corridor`.
- **Planned vs. Unplanned Resolution Schema Mismatch:** 
  - **Planned** events (rallies, festivals, construction) predominantly use `end_datetime` for scheduled resolution (97% populated, only 7% have `closed_datetime`).
  - **Unplanned** events (breakdowns, water logging) record completion times in `closed_datetime` (40% populated, only 0.3% have `end_datetime`).
  - **Coalescing Fix:** We coalesce these fields (`resolved_time = closed_datetime.fillna(end_datetime)`) to avoid dropping planned events. This increased the number of usable planned events in the training data from 33 to 297, aligning the model with the hackathon's "planned event" theme.
- **Data Filtration Drop Statistics:**
  - About 57% of raw records (primarily active events that lack a closed or end resolution timestamp) are filtered out during training preprocessing because their resolved durations cannot be determined.
  - This leaves 43% (3,441 records) of the dataset available for training and evaluation.

## Fundamental Approach (Why this way and not hit-and-trial)
Because we lack a direct `Y` variable for impact or resources, a naïve hit-and-trial approach would attempt to blindly map strings to outputs. Instead, we must use a robust, deterministic proxy approach:

### 1. Target Leakage Prevention & Downstream Policy Formulation (The "Why")
In previous versions, training the model directly on a synthetic target variable (`Impact_Score`) that was calculated using `priority_score` and `closure_multiplier` as features led to severe **Target Leakage**. Since the target was directly constructed using those two input features, the model easily reverse-engineered the formula, leading to artificially low errors but weak real-world predictive value.

To resolve this leakage:
1. **Machine Learning Model Target:** The machine learning model is trained to predict `np.log1p(duration_mins)` directly. Duration is a physical variable (the time to clear/resolve the event) and represents the true predictive challenge. The input features include `priority_score`, `closure_multiplier`, `event_type_planned`, temporal peak indicators, and spatial hotspots.
2. **Downstream Business Rules (Policy-driven Impact Score):** The final `Impact_Score` is computed downstream using the model's predicted duration:
   $$\text{raw\_impact} = \text{predicted\_duration} \times \text{priority\_score} \times \text{closure\_multiplier}$$
   $$\text{impact\_score} = 1.0 + 9.0 \times \frac{\log1p(\text{raw\_impact}) - \text{min\_val}}{\text{max\_val} - \text{min\_val}}$$
This ensures the model predictions are mathematically sound, leakage-free, and defendable in interviews.

### 2. Advanced Multi-Modal Feature Extraction (The "Why")
Instead of just feeding raw columns into a model, we extracted deep context:
- **Spatial Features:** We used **K-Means clustering** on Latitude/Longitude to automatically segment the city into 20 dynamic hotspot clusters. We then layered this with explicit geofencing by one-hot encoding the `zone` and `corridor` strings.
- **NLP Features:** The `description` column holds vital clues (e.g., "heavy", "blocked", "fallen"). We used **TF-IDF Vectorization** to turn these raw text descriptions into mathematical severity features.
- **Real-time Spatial-Temporal Load (`nearby_active_events`):** To capture the cumulative stress on the traffic network, we engineered a concurrent active incident feature. It calculates the number of other active, unresolved events within a 5 km radius at the start time of the event. This serves as a real-time signal of local network congestion.

### 3. Hyperparameter Tuned Voting Regressor (The "Why")
We utilized a **Voting Regressor**, combining the strengths of an **XGBoost Regressor** and a **RandomForest Regressor**.
- *Why not Deep Learning?* Tree-based models dominate tabular datasets of this scale (~8k rows). 
- *Why Tuning?* We optimized the XGBoost hyperparameters using `RandomizedSearchCV`. This identifies a highly performant parameter configuration within our search space, reducing overall variance compared to default configurations.

### 4. Explainable AI via SHAP (The "Why")
"Black box" AI is dangerous for law enforcement. We implemented **SHAP (SHapley Additive exPlanations)** to peek under the hood. The SHAP Summary Plot visibly demonstrates which features (like peak hour, a specific zone, or a TF-IDF text keyword) influence the model's predictions.
*Note: Because a VotingRegressor ensemble does not natively support TreeExplainer, we pass the RandomForest component (model.estimators_[1]) to SHAP as a proxy for the ensemble's feature importances.*

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


