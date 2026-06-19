import nbformat as nbf

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Data Loading Cell to include 'zone' and 'corridor'
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'cols_to_keep' in cell['source']:
        cell['source'] = cell['source'].replace(
            "'priority', 'veh_type', 'description'", 
            "'priority', 'veh_type', 'description', 'zone', 'corridor'"
        )

# 2. Add PIP install SHAP at the beginning
pip_cell = nbf.v4.new_code_cell("!pip install shap -q\nimport shap")
nb['cells'].insert(1, pip_cell)

# 3. Rewrite the Modeling Cell for V3 (Tuning + Zone/Corridor)
new_modeling_code = """import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Feature Engineering for Modeling

# A. Categorical Encoding (event_cause, veh_type, zone, corridor)
df['veh_type'] = df['veh_type'].fillna('unknown')
df['zone'] = df['zone'].fillna('unknown')
df['corridor'] = df['corridor'].fillna('unknown')

df_model = pd.get_dummies(df, columns=['event_cause', 'veh_type', 'zone', 'corridor'], drop_first=True)

# B. NLP on Description
print("Extracting NLP features from Incident Descriptions...")
df_model['description'] = df_model['description'].fillna('')
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
desc_tfidf = tfidf.fit_transform(df_model['description']).toarray()
tfidf_cols = [f'tfidf_{i}' for i in range(desc_tfidf.shape[1])]
df_tfidf = pd.DataFrame(desc_tfidf, columns=tfidf_cols, index=df_model.index)
df_model = pd.concat([df_model, df_tfidf], axis=1)

# Define Features (X) and Target (y)
base_features = ['hour', 'is_weekend', 'is_peak_hour', 'priority_score', 'closure_multiplier', 'hotspot_cluster']
cat_features = [c for c in df_model.columns if 'event_cause_' in c or 'veh_type_' in c or 'zone_' in c or 'corridor_' in c]
features = base_features + cat_features + tfidf_cols
           
X = df_model[features]
y = df_model['impact_score']

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning & Ensemble Training
print("Tuning Hyperparameters for XGBoost (this may take a minute)...")
xgb_base = xgb.XGBRegressor(random_state=42)
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}
random_search = RandomizedSearchCV(xgb_base, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
print(f"Best XGBoost Params: {random_search.best_params_}")

print("Training Advanced Ensemble Regressor (Tuned XGBoost + RandomForest)...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model = VotingRegressor([('xgb', best_xgb), ('rf', rf_model)])
model.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance -> RMSE: {rmse:.2f} | MAE: {mae:.2f}")
"""

# Replace the old Modeling cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'VotingRegressor' in cell['source'] and 'TfidfVectorizer' in cell['source']:
        nb['cells'][i]['source'] = new_modeling_code

# 4. Insert SHAP Cell after Evaluation Cell
shap_md = nbf.v4.new_markdown_cell("## Model Explainability (SHAP)\nTo build trust with the Traffic Police and Hackathon Judges, we use **SHAP (SHapley Additive exPlanations)** to peak inside the 'black box' and understand exactly *why* the AI predicts a severe congestion impact.")
shap_code = """import shap

# Initialize JavaScript visualizations in notebook
shap.initjs()

# Explain the XGBoost model predictions using SHAP
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# Plot the summary
print("Generating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
import matplotlib.pyplot as plt
plt.title("Top Feature Impacts on Congestion Severity")
plt.tight_layout()
plt.show()
"""
shap_cell = nbf.v4.new_code_cell(shap_code)

# Find where to insert SHAP (after the Model Evaluation Visualizations)
eval_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Resource Recommendation Matrix' in cell['source']:
        eval_idx = i
        break

nb['cells'].insert(eval_idx, shap_md)
nb['cells'].insert(eval_idx + 1, shap_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("V3 improvements applied successfully.")
