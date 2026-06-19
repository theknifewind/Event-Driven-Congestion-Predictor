import nbformat as nbf

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '!pip install shap' in cell['source']:
        cell['source'] = cell['source'].replace('!pip install shap -q', '!pip install shap "xgboost<2.1.0" -q')

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Fixed XGBoost version dependency in the notebook.")
