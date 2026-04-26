import nbformat as nbf
import os

def update_notebook_unified(filepath, model_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Identify common variable names used in the notebook
    all_source = "\n".join([c.source for c in nb.cells if c.cell_type == 'code'])
    
    preds_var = 'preds'
    if 'y_pred_xgb' in all_source: preds_var = 'y_pred_xgb'
    elif 'y_pred_svm' in all_source: preds_var = 'y_pred_svm'
    elif 'preds_knn' in all_source: preds_var = 'preds_knn'
    elif 'y_pred' in all_source: preds_var = 'y_pred'
    
    probs_var = 'probs'
    if 'y_proba_xgb' in all_source: probs_var = 'y_proba_xgb'
    elif 'y_proba_svm' in all_source: probs_var = 'y_proba_svm'
    elif 'probs_knn' in all_source: probs_var = 'probs_knn'
    elif 'probs2' in all_source: probs_var = 'probs2'
    elif 'y_proba' in all_source: probs_var = 'y_proba'

    # Special handling for Duel
    if 'Final_Champion_Duel' in filepath:
        preds_var = 'm2.predict(X_test_hybrid)'
        probs_var = 'm2.predict_proba(X_test_hybrid)[:, 1]'

    # 2. Evaluation code block
    eval_code = f"""
# --- STANDARDIZED EVALUATION ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, classification_report

try:
    final_preds = {preds_var}
    final_probs = {probs_var}
    
    print("\\n--- MODEL EVALUATION ---")
    print(classification_report(y_test, final_preds))

    roc_auc = roc_auc_score(y_test, final_probs)
    precision, recall, _ = precision_recall_curve(y_test, final_probs)
    pr_auc = auc(recall, precision)
    print(f"ROC-AUC: {{roc_auc:.4f}}")
    print(f"PR-AUC: {{pr_auc:.4f}}")

    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
except Exception as e:
    print(f"Evaluation Error: {{e}}")
"""

    # 3. Find the last code cell and append the evaluation code to it
    # This ensures it runs in the same context/cell as the training
    last_code_cell = None
    for cell in nb.cells:
        if cell.cell_type == 'code' and '--- STANDARDIZED EVALUATION ---' not in cell.source:
            last_code_cell = cell
    
    # Remove any existing standalone eval cells
    nb.cells = [c for c in nb.cells if '--- MODEL EVALUATION ---' not in c.source and '--- STANDARDIZED EVALUATION ---' not in c.source]

    if last_code_cell:
        last_code_cell.source += eval_code

    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Unified and Updated {filepath}")

notebooks = {
    'notebooks/04_Logistic_Regression.ipynb': 'Logistic Regression',
    'notebooks/06_SVM.ipynb': 'Linear SVM',
    'notebooks/07_Random_Forest.ipynb': 'Random Forest',
    'notebooks/08_XGBoost.ipynb': 'XGBoost (Baseline)',
    'notebooks/09_KNN.ipynb': 'K-Nearest Neighbors',
    'notebooks/10_Advanced_Experiments.ipynb': 'Advanced Champion (XGBoost)',
    'notebooks/11_Final_Champion_Duel.ipynb': 'Final Champion Duel'
}

for path, name in notebooks.items():
    if os.path.exists(path):
        update_notebook_unified(path, name)
