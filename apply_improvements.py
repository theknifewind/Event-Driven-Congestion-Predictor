import nbformat as nbf

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Data Loading Cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'cols_to_keep' in cell['source']:
        # Replace the cols_to_keep definition to include description
        cell['source'] = cell['source'].replace(
            "'priority', 'veh_type'", 
            "'priority', 'veh_type', 'description'"
        )

# 2. Update the Modeling Cell
new_modeling_code = """import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Feature Engineering for Modeling

# A. Categorical Encoding (event_cause and veh_type)
df['veh_type'] = df['veh_type'].fillna('unknown')
df_model = pd.get_dummies(df, columns=['event_cause', 'veh_type'], drop_first=True)

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
cat_features = [c for c in df_model.columns if 'event_cause_' in c or 'veh_type_' in c]
features = base_features + cat_features + tfidf_cols
           
X = df_model[features]
y = df_model['impact_score']

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Advanced Ensemble Model (XGBoost + Random Forest)
print("Training Advanced Ensemble Regressor (XGBoost + RandomForest)...")
xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model = VotingRegressor([('xgb', xgb_model), ('rf', rf_model)])
model.fit(X_train, y_train)

# 3. Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance -> RMSE: {rmse:.2f} | MAE: {mae:.2f}")
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'xgb.XGBRegressor' in cell['source']:
        cell['source'] = new_modeling_code

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook improvements applied successfully.")
