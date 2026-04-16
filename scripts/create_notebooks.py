"""
Generates two Jupyter notebooks from the existing .py source files:
  notebooks/01_Preprocessing.ipynb
  notebooks/02_EDA.ipynb

Run from the project root:
    python3 scripts/create_notebooks.py
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR   = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Notebook helpers
# ─────────────────────────────────────────────────────────────────────────────

def nb_template():
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": []
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "id": f"md_{abs(hash(text[:40])) % 10**8:08x}",
        "metadata": {},
        "source": text
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "id": f"code_{abs(hash(code[:40])) % 10**8:08x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code
    }

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_notebook():
    nb = nb_template()
    cells = nb["cells"]

    cells.append(md_cell(
        "# 🔒 IEEE-CIS Fraud Detection — Advanced Data Preprocessing Pipeline\n\n"
        "> **Goal:** Load the raw transaction and identity CSVs, apply a rigorous, "
        "object-oriented preprocessing pipeline, and export a clean, analysis-ready "
        "`data/processed_train.csv`.\n\n"
        "### Pipeline Overview\n"
        "| Step | Transformer | Purpose |\n"
        "|------|------------|--------|\n"
        "| 1 | `reduce_mem_usage` | Downcast dtypes to shrink RAM foot-print by ~50 % |\n"
        "| 2 | `DataMerger` | Left-join Transaction + Identity; fix `id-XX` → `id_XX` |\n"
        "| 3 | `TimeFeatureExtractor` | Extract hour-of-day & day-of-week from `TransactionDT` |\n"
        "| 4 | `DropHighNulls` | Remove columns with > 85 % missing values |\n"
        "| 5 | `FrequencyEncoder` | Replace categories with their relative frequency |\n"
        "| 6 | Final cast | Ensure all remaining columns are numeric |\n"
        "| 7 | Save | Write `data/processed_train.csv` |\n"
    ))

    cells.append(md_cell(
        "## ⚙️ Step 0 — Imports & Dependencies\n\n"
        "We use **scikit-learn's** `BaseEstimator` + `TransformerMixin` interfaces so every "
        "transformation is stateful, reusable, and can be slotted into a `Pipeline` without "
        "causing data leakage."
    ))

    cells.append(code_cell(
        "import os\n"
        "import warnings\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.base import BaseEstimator, TransformerMixin\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# ── Paths ───────────────────────────────────────────────────────────\n"
        "BASE_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
        "DATA_DIR   = os.path.join(BASE_DIR, 'data')\n"
        "OUTPUT_CSV = os.path.join(DATA_DIR, 'processed_train.csv')\n"
        "print('Base directory:', BASE_DIR)\n"
        "print('Data directory:', DATA_DIR)\n"
    ))

    # ── Memory optimiser ──────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🗜️ Step 1 — Memory Optimisation (`reduce_mem_usage`)\n\n"
        "### What it does\n"
        "Iterates every numeric column and downcasts it to the smallest data type that "
        "can safely hold its range (e.g. `float64` → `float16`, `int64` → `int8`).\n\n"
        "### Why it's necessary\n"
        "The raw CSVs total **~1.5 GB**. Pandas defaults to 64-bit types for everything, "
        "often wasting 4× the RAM actually required. Downcasting typically achieves a "
        "**50-70 % memory reduction**, making local experimentation feasible on a standard laptop."
    ))

    cells.append(code_cell(
        "def reduce_mem_usage(df, verbose=True):\n"
        "    \"\"\"Downcast numeric columns to the minimum safe dtype.\"\"\"\n"
        "    numerics  = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n"
        "    start_mem = df.memory_usage(deep=True).sum() / 1024**2\n"
        "    for col in df.columns:\n"
        "        col_type = df[col].dtype\n"
        "        if col_type.name in numerics:\n"
        "            c_min, c_max = df[col].min(), df[col].max()\n"
        "            if str(col_type)[:3] == 'int':\n"
        "                for t in [np.int8, np.int16, np.int32, np.int64]:\n"
        "                    if np.iinfo(t).min < c_min and c_max < np.iinfo(t).max:\n"
        "                        df[col] = df[col].astype(t); break\n"
        "            else:\n"
        "                for t in [np.float16, np.float32, np.float64]:\n"
        "                    if np.finfo(t).min < c_min and c_max < np.finfo(t).max:\n"
        "                        df[col] = df[col].astype(t); break\n"
        "    end_mem = df.memory_usage(deep=True).sum() / 1024**2\n"
        "    if verbose:\n"
        "        print(f'  RAM: {start_mem:.1f} MB  →  {end_mem:.1f} MB  '\n"
        "              f'({100*(start_mem-end_mem)/start_mem:.1f} % reduction)')\n"
        "    return df\n"
    ))

    # ── DataMerger ────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🔀 Step 2 — Data Merger (`DataMerger`)\n\n"
        "### What it does\n"
        "Performs a **left join** of the Transaction table onto the Identity table using "
        "`TransactionID` as the key.  It also normalises the identity column names by "
        "replacing dashes with underscores (`id-01` → `id_01`).\n\n"
        "### Why it's necessary\n"
        "Fraud footprints rely heavily on device/network metadata stored only in the identity table. "
        "Not every transaction has identity data (left join preserves all transactions). "
        "The dash/underscore mismatch between train and test identity columns would silently "
        "break model inference at scoring time if left unfixed."
    ))

    cells.append(code_cell(
        "class DataMerger(BaseEstimator, TransformerMixin):\n"
        "    \"\"\"Left-join Transaction + Identity; standardise column names.\"\"\"\n"
        "    def __init__(self, identity_df):\n"
        "        self.identity_df = identity_df\n\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n\n"
        "    def transform(self, X):\n"
        "        print('  Merging Transaction + Identity ...')\n"
        "        id_df = self.identity_df.copy()\n"
        "        id_df.columns = [c.replace('-', '_') for c in id_df.columns]\n"
        "        merged = X.merge(id_df, on='TransactionID', how='left')\n"
        "        print(f'  Shape after merge: {merged.shape}')\n"
        "        return merged\n"
    ))

    # ── TimeFeatureExtractor ──────────────────────────────────────────────────
    cells.append(md_cell(
        "## ⏰ Step 3 — Temporal Feature Engineering (`TimeFeatureExtractor`)\n\n"
        "### What it does\n"
        "Derives **`Transaction_Hour`** (0–23) and **`Transaction_Day`** (0–6) "
        "from the raw `TransactionDT` column, which is a timedelta in seconds from an "
        "unknown reference date.\n\n"
        "### Why it's necessary\n"
        "Raw seconds are a monotonically increasing counter that a tree model cannot "
        "easily convert into meaningful cyclical patterns. Deriving hour-of-day and "
        "day-of-week directly exposes temporal fraud signatures — e.g. card-testing "
        "spikes at 3 AM, or weekend automated attacks — with near-zero computational cost."
    ))

    cells.append(code_cell(
        "class TimeFeatureExtractor(BaseEstimator, TransformerMixin):\n"
        "    \"\"\"Extract hour-of-day and day-of-week from TransactionDT.\"\"\"\n"
        "    def fit(self, X, y=None):\n"
        "        return self\n\n"
        "    def transform(self, X):\n"
        "        print('  Extracting temporal features ...')\n"
        "        X = X.copy()\n"
        "        X['Transaction_Hour'] = (X['TransactionDT'] // 3600) % 24\n"
        "        X['Transaction_Day']  = (X['TransactionDT'] // 86400) % 7\n"
        "        return X\n"
    ))

    # ── DropHighNulls ─────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🗑️ Step 4 — High-Null Column Pruning (`DropHighNulls`)\n\n"
        "### What it does\n"
        "During **`fit`**, learns which columns exceed the null threshold.  "
        "During **`transform`**, drops those columns from any DataFrame passed in.\n\n"
        "### Why it's necessary\n"
        "Many of the `V`-series columns have > 90 % missing values. Imputing a column "
        "that is 95 % NaN forces the model to learn the imputed constant rather than a "
        "real signal. It also inflates dimensionality, worsening the *curse of "
        "dimensionality* for distance-based metrics. Dropping them is both statistically "
        "sound and computationally efficient.\n\n"
        "> **Threshold choice:** 0.85 (drop if > 85 % missing).  "
        "Adjust `threshold=` to trade coverage vs. noise."
    ))

    cells.append(code_cell(
        "class DropHighNulls(BaseEstimator, TransformerMixin):\n"
        "    \"\"\"Drop columns whose missing-value fraction exceeds `threshold`.\"\"\"\n"
        "    def __init__(self, threshold=0.85):\n"
        "        self.threshold = threshold\n"
        "        self.cols_to_drop_ = []\n\n"
        "    def fit(self, X, y=None):\n"
        "        null_frac = X.isnull().mean()\n"
        "        self.cols_to_drop_ = null_frac[null_frac > self.threshold].index.tolist()\n"
        "        return self\n\n"
        "    def transform(self, X):\n"
        "        print(f'  Dropping {len(self.cols_to_drop_)} high-null columns ...')\n"
        "        return X.drop(columns=self.cols_to_drop_, errors='ignore')\n"
    ))

    # ── FrequencyEncoder ──────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🔡 Step 5 — Frequency Encoding (`FrequencyEncoder`)\n\n"
        "### What it does\n"
        "Replaces each category in an object/categorical column with its **normalised "
        "frequency** (i.e. proportion of rows containing that value).  Frequencies are "
        "fitted on the training set and reused on the test set to prevent leakage.\n\n"
        "### Why it's necessary\n"
        "* **Label Encoding** assigns arbitrary integers — misleads ordinal-sensitive algorithms.\n"
        "* **One-Hot Encoding** creates hundreds of sparse binary columns — memory explosion.\n"
        "* **Frequency Encoding** produces a single meaningful numeric per category: rare "
        "  device fingerprints or unusual email domains get small values, which tree models "
        "  can split on precisely to detect anomalous, low-frequency fraud patterns."
    ))

    cells.append(code_cell(
        "class FrequencyEncoder(BaseEstimator, TransformerMixin):\n"
        "    \"\"\"Encode categories as their normalised frequency in the training set.\"\"\"\n"
        "    def __init__(self, cat_cols=None):\n"
        "        self.cat_cols = cat_cols\n"
        "        self.freq_maps_ = {}\n\n"
        "    def fit(self, X, y=None):\n"
        "        cols = self.cat_cols or X.select_dtypes(include=['object','category']).columns.tolist()\n"
        "        for col in cols:\n"
        "            if col in X.columns:\n"
        "                self.freq_maps_[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()\n"
        "        return self\n\n"
        "    def transform(self, X):\n"
        "        print(f'  Frequency-encoding {len(self.freq_maps_)} categorical columns ...')\n"
        "        X = X.copy()\n"
        "        for col, fmap in self.freq_maps_.items():\n"
        "            if col in X.columns:\n"
        "                X[col] = X[col].map(fmap).fillna(0)\n"
        "        return X\n"
    ))

    # ── Load & Run ────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 📂 Step 6 — Load Raw Data & Run the Full Pipeline\n\n"
        "Now that all transformers are defined we:\n"
        "1. Read both CSVs with `read_csv`.\n"
        "2. Immediately call `reduce_mem_usage` on both (before the pipeline, to save peak RAM).\n"
        "3. Assemble and execute the scikit-learn `Pipeline`.\n"
        "4. Cast any residual object columns to numeric.\n"
        "5. Persist to `data/processed_train.csv`."
    ))

    cells.append(code_cell(
        "print('[1/5] Loading raw CSVs ...')\n"
        "train_tx  = pd.read_csv(os.path.join(DATA_DIR, 'train_transaction.csv'))\n"
        "train_id  = pd.read_csv(os.path.join(DATA_DIR, 'train_identity.csv'))\n"
        "print(f'  train_transaction : {train_tx.shape}')\n"
        "print(f'  train_identity    : {train_id.shape}')\n\n"
        "print('\\n[2/5] Optimising memory ...')\n"
        "train_tx = reduce_mem_usage(train_tx)\n"
        "train_id = reduce_mem_usage(train_id)\n\n"
        "print('\\n[3/5] Running preprocessing pipeline ...')\n"
        "pipeline = Pipeline([\n"
        "    ('merger',       DataMerger(identity_df=train_id)),\n"
        "    ('time_extract', TimeFeatureExtractor()),\n"
        "    ('drop_nulls',   DropHighNulls(threshold=0.85)),\n"
        "    ('freq_encoder', FrequencyEncoder()),\n"
        "])\n"
        "processed = pipeline.fit_transform(train_tx)\n\n"
        "print('\\n[4/5] Final numeric cast ...')\n"
        "for col in processed.select_dtypes(include=['object']).columns:\n"
        "    processed[col] = pd.to_numeric(processed[col], errors='coerce')\n"
        "print(f'  Final shape : {processed.shape}')\n"
        "print(f'  isFraud distribution:\\n{processed[\"isFraud\"].value_counts()}')\n\n"
        "print(f'\\n[5/5] Saving to {OUTPUT_CSV} ...')\n"
        "processed.to_csv(OUTPUT_CSV, index=False)\n"
        "print('\\n✅  preprocessing complete!  processed_train.csv is ready.')\n"
    ))

    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────

def build_eda_notebook():
    nb = nb_template()
    cells = nb["cells"]

    cells.append(md_cell(
        "# 📊 IEEE-CIS Fraud Detection — Exploratory Data Analysis\n\n"
        "> **Input :** `data/processed_train.csv` — output of the preprocessing notebook.\n\n"
        "> **Outputs:** Publication-quality dark-mode figures saved to `reports/figures/` "
        "and a plain-text summary in `reports/eda_summary.txt`.\n\n"
        "### Sections\n"
        "| # | Topic |\n"
        "|---|-------|\n"
        "| 1 | Dataset overview & null audit |\n"
        "| 2 | Class imbalance (bar + pie) |\n"
        "| 3 | Transaction amount distributions |\n"
        "| 4 | Temporal fraud patterns (hour / day) |\n"
        "| 5 | Correlation heatmap — top 25 features |\n"
        "| 6 | Top-4 feature KDE: fraud vs legitimate |\n"
        "| 7 | Fraud rate by amount bucket |\n"
    ))

    # ── Setup ─────────────────────────────────────────────────────────────────
    cells.append(md_cell("## ⚙️ 0 — Setup: Imports, Paths & Style"))

    cells.append(code_cell(
        "import os, warnings\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.ticker as mticker\n"
        "import matplotlib.gridspec as gridspec\n"
        "import seaborn as sns\n"
        "warnings.filterwarnings('ignore')\n"
        "%matplotlib inline\n\n"
        "BASE_DIR    = os.path.abspath(os.path.join(os.getcwd(), '..'))\n"
        "DATA_PATH   = os.path.join(BASE_DIR, 'data', 'processed_train.csv')\n"
        "FIG_DIR     = os.path.join(BASE_DIR, 'reports', 'figures')\n"
        "REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'eda_summary.txt')\n"
        "os.makedirs(FIG_DIR, exist_ok=True)\n"
        "os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)\n\n"
        "# ── Dark-mode style ──────────────────────────────────────────────────\n"
        "BG      = '#0F1117'; FRAUD   = '#E74C3C'; LEGIT = '#4A90D9'\n"
        "ACCENT  = '#F39C12'; TEXT    = '#EAEAEA'; GRID  = '#2A2D3E'\n"
        "plt.rcParams.update({\n"
        "    'figure.facecolor': BG,  'axes.facecolor': BG,\n"
        "    'axes.edgecolor':   GRID,'axes.labelcolor': TEXT,\n"
        "    'axes.titlecolor':  TEXT,'xtick.color':     TEXT,\n"
        "    'ytick.color':      TEXT,'grid.color':      GRID,\n"
        "    'text.color':       TEXT,'legend.facecolor':'#1A1D2E',\n"
        "    'legend.edgecolor': GRID,'font.size':        11,\n"
        "    'axes.titlesize':   14,  'axes.labelsize':   12,\n"
        "})\n\n"
        "def savefig(fig, name):\n"
        "    path = os.path.join(FIG_DIR, name)\n"
        "    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=BG)\n"
        "    print(f'  Figure saved → reports/figures/{name}')\n"
        "    plt.show()\n\n"
        "print('Setup complete.')\n"
    ))

    cells.append(code_cell(
        "print('Loading processed dataset ...')\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "print(f'Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns')\n"
        "df.head(3)\n"
    ))

    # ── Section 1 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🗂️ 1 — Dataset Overview\n\n"
        "### What\n"
        "Print basic shape, dtype breakdown, memory footprint, and any residual missing values.\n\n"
        "### Why\n"
        "Confirms the preprocessing pipeline achieved its goals: nulls should be minimal, "
        "dtypes compact, and all object columns should have been encoded."
    ))

    cells.append(code_cell(
        "print(f'Shape          : {df.shape}')\n"
        "print(f'Memory         : {df.memory_usage(deep=True).sum()/1024**2:.1f} MB')\n"
        "print(f'Numeric cols   : {df.select_dtypes(include=\"number\").shape[1]}')\n"
        "print(f'Object cols    : {df.select_dtypes(include=\"object\").shape[1]}')\n\n"
        "nan_summary = df.isnull().sum()\n"
        "nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False).head(15)\n"
        "if nan_summary.empty:\n"
        "    print('\\n✅  No missing values remain!')\n"
        "else:\n"
        "    print('\\nRemaining NaNs (top 15):')\n"
        "    for col, cnt in nan_summary.items():\n"
        "        print(f'  {col:<35} {cnt:>8,}  ({100*cnt/len(df):.1f}%)')\n"
    ))

    # ── Section 2 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## ⚖️ 2 — Class Imbalance Analysis\n\n"
        "### What\n"
        "Bar chart + pie chart showing the ratio of legitimate to fraudulent transactions.\n\n"
        "### Why\n"
        "Fraud datasets are almost always severely imbalanced (~3 % fraud). This drives "
        "critical decisions:\n"
        "- **Accuracy** becomes a misleading metric (always-predict-legit wins at 97 %).\n"
        "- We must optimise **AUC-PR, F1, or ROC-AUC** instead.\n"
        "- Training may require **class-weight balancing** or **SMOTE oversampling**."
    ))

    cells.append(code_cell(
        "counts = df['isFraud'].value_counts()\n"
        "pct_f  = 100 * counts[1] / len(df)\n"
        "pct_l  = 100 * counts[0] / len(df)\n"
        "print(f'Legitimate : {counts[0]:,}  ({pct_l:.2f} %)')\n"
        "print(f'Fraudulent : {counts[1]:,}  ({pct_f:.2f} %)')\n"
        "print(f'Imbalance  : {counts[0]/counts[1]:.1f}:1')\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n"
        "fig.suptitle('Class Distribution — Fraudulent vs Legitimate', fontsize=16, y=1.01)\n\n"
        "bars = ax1.bar(['Legitimate', 'Fraudulent'], [counts[0], counts[1]],\n"
        "               color=[LEGIT, FRAUD], width=0.5, edgecolor='none')\n"
        "ax1.set_title('Transaction Counts'); ax1.set_ylabel('Count')\n"
        "ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))\n"
        "for bar, cnt, pct in zip(bars,[counts[0],counts[1]],[pct_l,pct_f]):\n"
        "    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2000,\n"
        "             f'{cnt:,}\\n({pct:.1f}%)', ha='center', color=TEXT, fontweight='bold')\n"
        "ax1.set_ylim(0, counts[0]*1.2)\n\n"
        "wedges,texts,autos = ax2.pie([counts[0],counts[1]], labels=['Legitimate','Fraudulent'],\n"
        "    colors=[LEGIT, FRAUD], autopct='%1.2f%%', startangle=140,\n"
        "    wedgeprops=dict(edgecolor=BG, linewidth=2), pctdistance=0.75)\n"
        "for t in autos: t.set_color(TEXT); t.set_fontsize(12)\n"
        "ax2.set_title('Proportion')\n\n"
        "fig.tight_layout()\n"
        "savefig(fig, '02_class_imbalance.png')\n"
    ))

    # ── Section 3 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 💵 3 — Transaction Amount Distributions\n\n"
        "### What\n"
        "KDE (density) and log-scale box plots of `TransactionAmt` split by fraud label.\n\n"
        "### Why\n"
        "Fraudsters exhibit characteristic amount patterns:\n"
        "- **Low-value card testing** — tiny transactions to check if a stolen card is live.\n"
        "- **Round-number fraud** — automated tools often transact at suspiciously round amounts.\n"
        "- **Log scale** is necessary because transaction amounts span several orders of magnitude."
    ))

    cells.append(code_cell(
        "legit_amt = df.loc[df['isFraud']==0, 'TransactionAmt'].dropna()\n"
        "fraud_amt = df.loc[df['isFraud']==1, 'TransactionAmt'].dropna()\n\n"
        "for label, s in [('Legitimate', legit_amt), ('Fraudulent', fraud_amt)]:\n"
        "    print(f'{label}: mean={s.mean():.2f}  median={s.median():.2f}  '\n"
        "          f'std={s.std():.2f}  max={s.max():.2f}')\n\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
        "fig.suptitle('Transaction Amount Distribution by Fraud Label', fontsize=16)\n\n"
        "p99 = df['TransactionAmt'].quantile(0.99)\n"
        "sns.kdeplot(legit_amt.clip(upper=p99), ax=ax1, color=LEGIT,\n"
        "            label='Legitimate', fill=True, alpha=0.35, linewidth=2)\n"
        "sns.kdeplot(fraud_amt.clip(upper=p99), ax=ax1, color=FRAUD,\n"
        "            label='Fraudulent', fill=True, alpha=0.50, linewidth=2)\n"
        "ax1.set_xlabel('TransactionAmt (clipped 99th pct)')\n"
        "ax1.set_ylabel('Density'); ax1.set_title('KDE — Linear Scale'); ax1.legend()\n\n"
        "plot_df = pd.DataFrame({\n"
        "    'Amount': pd.concat([legit_amt, fraud_amt]),\n"
        "    'Label': ['Legitimate']*len(legit_amt) + ['Fraudulent']*len(fraud_amt),\n"
        "})\n"
        "sns.boxplot(data=plot_df, x='Label', y='Amount', palette=[LEGIT, FRAUD],\n"
        "            ax=ax2, linewidth=1.5,\n"
        "            flierprops=dict(marker='.', color=ACCENT, markersize=2, alpha=0.3))\n"
        "ax2.set_yscale('log'); ax2.set_title('Box Plot — Log Scale')\n"
        "ax2.set_xlabel(''); ax2.set_ylabel('TransactionAmt (log)')\n\n"
        "fig.tight_layout(); savefig(fig, '03_transaction_amount.png')\n"
    ))

    # ── Section 4 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## ⏱️ 4 — Temporal Fraud Patterns\n\n"
        "### What\n"
        "Bar charts of fraud rate grouped by `Transaction_Hour` and `Transaction_Day` — "
        "features we engineered in the preprocessing step.\n\n"
        "### Why\n"
        "Fraud is not uniformly distributed over time:\n"
        "- **Hour of day:** fraudulent bots often run at off-peak hours (late night / early morning).\n"
        "- **Day of week:** some fraud patterns peak on weekdays when bank monitoring is lighter.\n\n"
        "This validates that our time-feature engineering added genuine signal."
    ))

    cells.append(code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "fig.suptitle('Temporal Fraud Patterns', fontsize=16)\n\n"
        "for ax, col, title, xlabel in zip(\n"
        "    axes,\n"
        "    ['Transaction_Hour', 'Transaction_Day'],\n"
        "    ['Fraud Rate by Hour of Day', 'Fraud Rate by Day of Week'],\n"
        "    ['Hour (0 = midnight)', 'Day (0 = Mon, 6 = Sun)'],\n"
        "):\n"
        "    if col not in df.columns:\n"
        "        ax.set_visible(False); continue\n"
        "    grp = df.groupby(col)['isFraud'].mean().reset_index()\n"
        "    grp.columns = [col, 'fraud_rate']\n"
        "    ax.bar(grp[col], grp['fraud_rate']*100, color=FRAUD, edgecolor='none', alpha=0.85)\n"
        "    ax.axhline(df['isFraud'].mean()*100, color=ACCENT, linestyle='--',\n"
        "               linewidth=1.5, label='Overall avg')\n"
        "    ax.set_xlabel(xlabel); ax.set_ylabel('Fraud Rate (%)')\n"
        "    ax.set_title(title); ax.legend()\n\n"
        "fig.tight_layout(); savefig(fig, '04_temporal_patterns.png')\n"
    ))

    # ── Section 5 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🔥 5 — Correlation Heatmap\n\n"
        "### What\n"
        "Compute Pearson |ρ| of every numeric feature against `isFraud`, select the top 25, "
        "and display a triangular correlation heatmap.\n\n"
        "### Why\n"
        "- Identifies the most linearly discriminative features before any model training.\n"
        "- Reveals multicollinearity between features (e.g. C-series counters), helping "
        "  prune redundant features and speed up training.\n"
        "- Acts as a fast, model-agnostic feature importance proxy."
    ))

    cells.append(code_cell(
        "num_df = df.select_dtypes(include='number')\n"
        "corr_target = num_df.corr()['isFraud'].drop('isFraud').abs().sort_values(ascending=False)\n"
        "top25 = corr_target.head(25).index.tolist()\n\n"
        "print('Top 10 features by |Pearson ρ| with isFraud:')\n"
        "for feat, val in corr_target.head(10).items():\n"
        "    print(f'  {feat:<35} {val:.4f}')\n\n"
        "corr_matrix = num_df[top25 + ['isFraud']].corr()\n"
        "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n\n"
        "fig, ax = plt.subplots(figsize=(16, 13))\n"
        "sns.heatmap(corr_matrix, mask=mask, ax=ax, cmap='coolwarm',\n"
        "            center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor=BG,\n"
        "            annot=(len(corr_matrix) <= 20), fmt='.2f',\n"
        "            annot_kws={'size': 7},\n"
        "            cbar_kws={'shrink': 0.75, 'label': 'Pearson ρ'})\n"
        "ax.set_title('Correlation Matrix — Top 25 Features + isFraud', fontsize=15, pad=15)\n"
        "plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)\n"
        "fig.tight_layout(); savefig(fig, '05_correlation_heatmap.png')\n"
    ))

    # ── Section 6 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 📐 6 — Top Feature Distributions: Fraud vs Legitimate\n\n"
        "### What\n"
        "KDE plots for the **4 most correlated features** showing the distribution "
        "separately for fraudulent and legitimate transactions.\n\n"
        "### Why\n"
        "Correlation tells us *that* a feature is predictive; this plot shows us *how* — "
        "i.e. whether fraud is concentrated in the tails, bimodal, or uniformly shifted. "
        "This informs threshold choices and whether transformations (log, clipping) would help."
    ))

    cells.append(code_cell(
        "top4 = [c for c in corr_target.head(10).index\n"
        "        if c not in ('TransactionID', 'TransactionDT')][:4]\n"
        "print('Top-4 features selected:', top4)\n\n"
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
        "axes = axes.flatten()\n"
        "fig.suptitle('Feature Distributions — Fraud vs Legitimate', fontsize=16)\n\n"
        "for ax, col in zip(axes, top4):\n"
        "    p01, p99 = df[col].quantile([0.01, 0.99])\n"
        "    legit = df.loc[df['isFraud']==0, col].dropna().clip(p01, p99)\n"
        "    fraud = df.loc[df['isFraud']==1, col].dropna().clip(p01, p99)\n"
        "    sns.kdeplot(legit, ax=ax, color=LEGIT, label='Legitimate', fill=True, alpha=0.30, linewidth=2)\n"
        "    sns.kdeplot(fraud, ax=ax, color=FRAUD, label='Fraudulent', fill=True, alpha=0.55, linewidth=2)\n"
        "    ax.set_title(col, fontsize=11); ax.set_xlabel(''); ax.legend(fontsize=9)\n\n"
        "fig.tight_layout(); savefig(fig, '06_topfeature_distributions.png')\n"
    ))

    # ── Section 7 ─────────────────────────────────────────────────────────────
    cells.append(md_cell(
        "## 🪣 7 — Fraud Rate by Transaction Amount Bucket\n\n"
        "### What\n"
        "Bin `TransactionAmt` into eight intuitive ranges and overlay the fraud rate "
        "(line) against the transaction count (bars) on a dual-axis chart.\n\n"
        "### Why\n"
        "Fraud risk is not monotonic across transaction sizes:\n"
        "- **< \\$10** — classic card-testing fraud (very high fraud rate).\n"
        "- **\\$1 K–\\$5 K** — high-value automated fraud.\n"
        "- **Mid-range \\$100–\\$500** — legitimate purchases dominate, low fraud rate.\n\n"
        "This plot directly motivates the use of `TransactionAmt` as a key feature and "
        "potentially as an interaction term in the final model."
    ))

    cells.append(code_cell(
        "bins   = [0, 10, 50, 100, 250, 500, 1000, 5000, float('inf')]\n"
        "labels = ['<$10','$10-50','$50-100','$100-250','$250-500','$500-1K','$1K-5K','>$5K']\n\n"
        "df2 = df[['TransactionAmt','isFraud']].dropna().copy()\n"
        "df2['AmtBucket'] = pd.cut(df2['TransactionAmt'], bins=bins, labels=labels)\n"
        "bucket = df2.groupby('AmtBucket', observed=True)['isFraud'].agg(['mean','count']).reset_index()\n"
        "bucket.columns = ['AmtBucket','fraud_rate','n']\n\n"
        "print('Fraud rate by amount bucket:')\n"
        "for _, r in bucket.iterrows():\n"
        "    print(f'  {str(r.AmtBucket):<12}  {r.fraud_rate*100:5.2f}%   n={int(r.n):,}')\n\n"
        "fig, ax1 = plt.subplots(figsize=(13, 5))\n"
        "fig.suptitle('Fraud Rate & Volume by Transaction Amount Bucket', fontsize=15)\n"
        "x = range(len(bucket))\n\n"
        "ax2 = ax1.twinx()\n"
        "ax2.bar(x, bucket['n'], color=LEGIT, alpha=0.20, label='# Transactions')\n"
        "ax2.set_ylabel('Transaction Count', color=LEGIT)\n"
        "ax2.tick_params(axis='y', colors=LEGIT)\n"
        "ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{int(v):,}'))\n\n"
        "ax1.plot(x, bucket['fraud_rate']*100, color=FRAUD, marker='o',\n"
        "         linewidth=2.5, markersize=8, label='Fraud Rate %', zorder=5)\n"
        "ax1.axhline(df['isFraud'].mean()*100, color=ACCENT, linestyle='--',\n"
        "            linewidth=1.2, label='Overall avg', zorder=4)\n"
        "ax1.set_ylabel('Fraud Rate (%)', color=FRAUD)\n"
        "ax1.tick_params(axis='y', colors=FRAUD)\n"
        "ax1.set_xticks(list(x)); ax1.set_xticklabels(bucket['AmtBucket'].astype(str), rotation=20, ha='right')\n"
        "ax1.set_xlabel('Transaction Amount Bucket')\n\n"
        "l1, labs1 = ax1.get_legend_handles_labels()\n"
        "l2, labs2 = ax2.get_legend_handles_labels()\n"
        "ax1.legend(l1+l2, labs1+labs2, loc='upper right')\n\n"
        "fig.tight_layout(); savefig(fig, '07_fraud_by_amount_bucket.png')\n"
        "print('\\n✅  All EDA sections complete!')\n"
    ))

    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks to disk
# ─────────────────────────────────────────────────────────────────────────────

def write_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✔ Written → {os.path.relpath(path, BASE_DIR)}")


if __name__ == "__main__":
    print("Building notebooks...")
    write_notebook(build_preprocessing_notebook(),
                   os.path.join(NB_DIR, "01_Preprocessing.ipynb"))
    write_notebook(build_eda_notebook(),
                   os.path.join(NB_DIR, "02_EDA.ipynb"))
    print("\nDone! Open the notebooks/ directory in Jupyter.")
