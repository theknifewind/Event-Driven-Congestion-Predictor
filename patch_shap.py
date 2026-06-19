import nbformat as nbf

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'shap.TreeExplainer' in cell['source']:
        cell['source'] = cell['source'].replace('shap.TreeExplainer(best_xgb)', 'shap.TreeExplainer(model.estimators_[1])')
        cell['source'] = cell['source'].replace('Explain the XGBoost model predictions', 'Explain the RandomForest model predictions (as a stable proxy for the ensemble)')

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Swapped XGBoost for RandomForest in SHAP Explainer.")
