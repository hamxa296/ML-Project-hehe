import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, confusion_matrix

def evaluate_model(y_test, probs, preds):
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds))

    pr, rc, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rc, pr)
    roc_auc = roc_auc_score(y_test, probs)
    print(f"PR-AUC: {pr_auc:.4f} (Paper XGBoost Benchmark: 0.834)")
    print(f"ROC-AUC: {roc_auc:.4f} (Paper XGBoost Benchmark: 0.887)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix — Paper-Exact Pipeline')
    plt.show()
