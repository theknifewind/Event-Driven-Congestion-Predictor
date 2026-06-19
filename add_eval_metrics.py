import nbformat as nbf

notebook_path = r"C:\Users\sriji\Projects\Event-Driven Congestion\Event_Driven_Congestion.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Find the index of the Resource Recommendation Markdown cell
insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Resource Recommendation Matrix' in cell['source']:
        insert_idx = i
        break

# Create new cells
eval_md = nbf.v4.new_markdown_cell("### Model Evaluation Visualizations\nTo ensure our model is trustworthy, we visualize its predictions against the actual historical impact scores.")

eval_code = """import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# 1. Actual vs Predicted Scatter Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='#2ecc71', edgecolor=None)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Impact Score', fontsize=12)
plt.xlabel('Actual Impact Score')
plt.ylabel('Predicted Impact Score')
plt.legend()

# 2. Residual (Error) Distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='#9b59b6', bins=30)
plt.title('Distribution of Prediction Errors (Residuals)', fontsize=12)
plt.xlabel('Error (Actual - Predicted)')
plt.axvline(0, color='red', linestyle='--', lw=2)

plt.tight_layout()
plt.show()
"""
eval_cell = nbf.v4.new_code_cell(eval_code)

# Insert the cells
nb['cells'].insert(insert_idx, eval_md)
nb['cells'].insert(insert_idx + 1, eval_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Evaluation metrics cells successfully inserted.")
