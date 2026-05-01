"""
association.py — Fraud-Linked Association Rule Mining
-----------------------------------------------------
Task: Find combinations of categorical transaction properties that
      co-occur significantly with fraud (high lift, high confidence).

Uses mlxtend FPGrowth + association_rules on discretized features.
Answers: "What card+product+email combinations signal fraud?"

Outputs:
  - artifacts/association_rules.json  (top-20 rules by lift)
  - results/graphs/latest_association_lift.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path


def _discretize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert relevant columns into descriptive string items for basket encoding.
    Uses columns that are most interpretable for fraud analysis.
    """
    d = pd.DataFrame()

    # ProductCD: direct categorical
    if 'ProductCD' in df.columns:
        d['ProductCD'] = 'ProductCD=' + df['ProductCD'].astype(str).str.strip()

    # card4 (visa/mastercard etc.) and card6 (debit/credit)
    for col in ['card4', 'card6']:
        if col in df.columns:
            d[col] = col + '=' + df[col].astype(str).str.strip()

    # P_emaildomain — keep top domains, group rest as 'other'
    if 'P_emaildomain' in df.columns:
        top_domains = df['P_emaildomain'].value_counts().nlargest(6).index.tolist()
        d['P_emaildomain'] = df['P_emaildomain'].apply(
            lambda x: f"email={x}" if x in top_domains else "email=other"
        )

    # TransactionAmt — bin into Low/Mid/High
    if 'TransactionAmt' in df.columns:
        amt = pd.to_numeric(df['TransactionAmt'], errors='coerce').fillna(0)
        bins = [0, 50, 200, float('inf')]
        labels = ['Amt=Low', 'Amt=Mid', 'Amt=High']
        d['AmtBin'] = pd.cut(amt, bins=bins, labels=labels, right=True).astype(str)

    # isFraud as an item so we can mine rules → fraud
    d['isFraud'] = df['isFraud'].map({0: 'isFraud=No', 1: 'isFraud=Yes'})

    return d.fillna('Unknown')


def run_association(train_df: pd.DataFrame, artifacts_dir: Path, graphs_dir: Path,
                    min_support: float = 0.01, min_confidence: float = 0.05,
                    top_n: int = 20) -> dict:
    """
    Association rule mining for fraud patterns:
      1. Discretize categorical transaction features
      2. One-hot encode (transaction basket)
      3. Run FPGrowth to find frequent itemsets
      4. Generate association rules, filter by confidence
      5. Sort by lift — highest lift = strongest non-random association
      6. Save top-N rules as JSON + bar chart of top rules by lift

    Returns dict with top rules.
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules as mlxtend_ar
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        print(">>> [Association] mlxtend not installed — skipping. Add 'mlxtend' to requirements.txt")
        return {"error": "mlxtend not installed", "rules": []}

    artifacts_dir = Path(artifacts_dir)
    graphs_dir = Path(graphs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("\n>>> [Association] Discretizing features...")
    disc = _discretize(train_df)

    # Sample to keep FPGrowth tractable (472K rows is slow)
    sample_size = min(80_000, len(disc))
    disc_sample = disc.sample(sample_size, random_state=42)
    print(f">>> [Association] Running FPGrowth on {sample_size:,} transactions | "
          f"min_support={min_support}")

    # TransactionEncoder expects list-of-lists
    records = disc_sample.astype(str).values.tolist()
    te = TransactionEncoder()
    te_array = te.fit_transform(records)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    print(f">>> [Association] {len(frequent_itemsets)} frequent itemsets found")

    if len(frequent_itemsets) == 0:
        print(">>> [Association] No itemsets found — try lowering min_support")
        return {"rules": [], "n_itemsets": 0}

    rules = mlxtend_ar(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    # Focus on rules that CONSEQUENT is isFraud=Yes
    fraud_rules = rules[rules['consequents'].apply(lambda x: 'isFraud=Yes' in x)].copy()
    fraud_rules = fraud_rules.sort_values('lift', ascending=False).head(top_n)

    print(f">>> [Association] {len(fraud_rules)} fraud-consequent rules | "
          f"Top lift: {fraud_rules['lift'].max():.3f}")

    def fmt_frozenset(fs):
        return list(fs)

    rules_list = []
    for _, row in fraud_rules.iterrows():
        rules_list.append({
            "antecedents":  fmt_frozenset(row['antecedents']),
            "consequents":  fmt_frozenset(row['consequents']),
            "support":      round(float(row['support']), 5),
            "confidence":   round(float(row['confidence']), 4),
            "lift":         round(float(row['lift']), 4),
        })

    result = {
        "task":            "Association Rule Mining (FPGrowth)",
        "sample_size":     sample_size,
        "min_support":     min_support,
        "min_confidence":  min_confidence,
        "n_itemsets":      int(len(frequent_itemsets)),
        "n_fraud_rules":   int(len(fraud_rules)),
        "top_rules":       rules_list,
    }

    out_path = artifacts_dir / 'association_rules.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f">>> [Association] Rules saved → {out_path}")

    _save_lift_chart(rules_list, graphs_dir)
    return result


def _save_lift_chart(rules_list: list, graphs_dir: Path):
    if not rules_list:
        return
    PALETTE = {'bg':'#0d1117','card':'#161b22','emerald':'#10b981','cyan':'#00d4ff',
               'muted':'#6b7280','border':'#30363d','text':'#e6edf3'}
    plt.style.use('dark_background')

    top = rules_list[:15]  # plot top 15
    labels = [' + '.join(r['antecedents']) for r in top]
    lifts  = [r['lift'] for r in top]
    confs  = [r['confidence'] * 100 for r in top]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor(PALETTE['bg']); ax.set_facecolor(PALETTE['card'])
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, lifts, color=PALETTE['emerald'], edgecolor=PALETTE['border'],
                   linewidth=0.7, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=PALETTE['text'])
    ax.set_xlabel('Lift (higher = stronger fraud association)', color=PALETTE['muted'], fontsize=11)
    ax.set_title('Top Association Rules → isFraud=Yes (by Lift)',
                 color=PALETTE['text'], fontsize=14, fontweight='bold', pad=14)
    ax.axvline(1.0, color=PALETTE['muted'], linestyle='--', linewidth=1,
               label='Lift=1 (random)')
    ax.tick_params(colors=PALETTE['muted'])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE['border'])
    ax.grid(True, axis='x', color=PALETTE['border'], linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['border'],
              labelcolor=PALETTE['text'], fontsize=10)
    plt.tight_layout()
    out = graphs_dir / 'latest_association_lift.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f">>> [Association] Lift chart → {out}")
