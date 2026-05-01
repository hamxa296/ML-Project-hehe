Paper-style reproduction for IEEE-CIS Fraud Detection

Steps:
- Build paper-style features (frequency, aggregations, PCA on V columns)
- CV target-encoding for categorical features
- Train LightGBM if available, otherwise XGBoost
- Produce OOF predictions and a stacked meta-logistic model

Run:
python run_paper_repl.py --sample 200000
