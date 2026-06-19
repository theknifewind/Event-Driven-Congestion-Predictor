import nbformat as nbf
import json

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if 'squared=False' in cell['source']:
            # Replace the rmse line
            cell['source'] = cell['source'].replace(
                'rmse = mean_squared_error(y_test, y_pred, squared=False)',
                'import numpy as np\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))'
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Fixed RMSE calculation in notebook.")
