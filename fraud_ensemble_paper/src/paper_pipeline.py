from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"
RANDOM_STATE = 42


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET, TIME_COL, ID_COL} and df[c].dtype == "object"]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET, TIME_COL, ID_COL} and pd.api.types.is_numeric_dtype(df[c])]


def _v_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("V")]


def _frequency_encode(series: pd.Series) -> pd.Series:
    counts = series.value_counts(dropna=False)
    return series.map(counts).fillna(0).astype(np.float32)


def _safe_group_agg(df: pd.DataFrame, group_col: str, agg_cols: Iterable[str]) -> pd.DataFrame:
    present = [c for c in agg_cols if c in df.columns]
    if not present:
        return pd.DataFrame(index=df.index)
    grouped = df.groupby(group_col, dropna=False)[present].agg(["mean", "std", "count"])
    grouped.columns = [f"{group_col}__{col[0]}_{col[1]}" for col in grouped.columns]
    return df[[group_col]].merge(grouped, left_on=group_col, right_index=True, how="left").drop(columns=[group_col])


@dataclass
class PaperPreprocessor:
    missing_threshold: float = 0.95
    corr_threshold: float = 0.98
    mi_threshold: float = 0.001
    pca_components: int = 5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PaperPreprocessor":
        # Step 1: Prune raw baseline features according to paper (missing -> zero-var -> corr -> MI)
        df_raw = X.copy()
        TARGET_BASELINE = 167

        missing_rate = df_raw.isna().mean()
        self.keep_missing_ = [c for c in df_raw.columns if missing_rate[c] <= self.missing_threshold]
        df_pruned = df_raw[self.keep_missing_]

        # Work on numeric baseline columns for variance/correlation/MI pruning
        numeric_baseline = df_pruned.select_dtypes(include=[np.number]).copy()

        # Zero-variance filter
        nz = numeric_baseline.nunique(dropna=False) > 1
        self.keep_zero_var_ = list(nz[nz].index)
        numeric_baseline = numeric_baseline[self.keep_zero_var_]

        # Correlation filter
        corr = numeric_baseline.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_corr = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        self.keep_corr_ = [c for c in numeric_baseline.columns if c not in drop_corr]
        numeric_baseline = numeric_baseline[self.keep_corr_]

        # Mutual information filter: compute MI across remaining numeric and categorical features
        # Prepare dataframe for MI computation: include numeric_baseline and encode categorical baseline
        cat_cols = [c for c in df_pruned.columns if c not in numeric_baseline.columns]
        cat_encoded = pd.DataFrame(
            {
                c: pd.factorize(df_pruned[c].fillna("__missing__").astype(str))[0]
                for c in cat_cols
            },
            index=df_pruned.index,
        )

        mi_df = pd.concat([numeric_baseline.fillna(0), cat_encoded.fillna(0)], axis=1)
        if mi_df.shape[1] > 0:
            mi = mutual_info_classif(mi_df, y.astype(int), random_state=RANDOM_STATE)
            self.mi_scores_ = pd.Series(mi, index=mi_df.columns)
            ranked = self.mi_scores_.sort_values(ascending=False)
            # Keep features above threshold, then pad/truncate to the paper's target baseline count.
            keep_mi = list(ranked[ranked >= self.mi_threshold].index)
            if len(keep_mi) < TARGET_BASELINE:
                keep_mi = list(ranked.index[: min(TARGET_BASELINE, len(ranked))])
            elif len(keep_mi) > TARGET_BASELINE:
                keep_mi = list(ranked.loc[keep_mi].index[:TARGET_BASELINE])
        else:
            self.mi_scores_ = pd.Series(dtype=float)
            keep_mi = []

        # Baseline feature lists
        self.keep_baseline_numeric_ = [c for c in keep_mi if c in numeric_baseline.columns]
        self.categorical_baseline_ = [c for c in keep_mi if c in cat_encoded.columns]

        # Build engineered features only on the pruned baseline set (paper order)
        baseline_cols = list(self.keep_baseline_numeric_) + list(self.categorical_baseline_)
        baseline_df = df_pruned[baseline_cols].copy() if baseline_cols else pd.DataFrame(index=df_pruned.index)
        engineered = self._build_features(baseline_df, fit=True)

        # Save final feature names and fit imputer on engineered numeric columns
        self.feature_names_ = list(engineered.columns)
        numeric_engineered = engineered.select_dtypes(include=[np.number]).copy()
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        if not numeric_engineered.empty:
            # fit imputer on engineered numeric features
            self.numeric_imputer_.fit(numeric_engineered)
            self.keep_numeric_ = list(numeric_engineered.columns)
        else:
            self.keep_numeric_ = []

        # For backward compatibility: set categorical/numeric lists
        self.categorical_columns_ = list(self.categorical_baseline_)
        self.numeric_columns_ = list(self.keep_baseline_numeric_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Build features on incoming X, then align to the feature names discovered at fit time
        df = self._build_features(X, fit=False)
        # Keep only the engineered feature columns discovered during fit
        cols = [c for c in self.feature_names_ if c in df.columns]
        df = df.reindex(columns=cols, fill_value=np.nan)

        # Impute numeric engineered features
        numeric = df.select_dtypes(include=[np.number]).copy()
        if hasattr(self, "numeric_imputer_") and not numeric.empty:
            numeric = pd.DataFrame(self.numeric_imputer_.transform(numeric), columns=numeric.columns, index=df.index)

        # Keep only the numeric engineered columns that were fitted
        if getattr(self, "keep_numeric_", None):
            for col in list(numeric.columns):
                if col not in self.keep_numeric_:
                    numeric = numeric.drop(columns=[col])
            return numeric[self.keep_numeric_].copy()

        return numeric.copy()

    def _build_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = X.copy()
        if fit:
            self.category_levels_ = {}

        # Keep the core signal and add a few paper-style engineered features.
        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
            df["TransactionAmt_missing"] = df["TransactionAmt"].isna().astype(np.float32)
        if "dist1" in df.columns:
            df["dist1_missing"] = df["dist1"].isna().astype(np.float32)
        if "dist2" in df.columns:
            df["dist2_missing"] = df["dist2"].isna().astype(np.float32)

        cat_cols = _categorical_columns(df)
        for col in cat_cols:
            if fit:
                self.category_levels_[col] = list(pd.Series(df[col].fillna("missing")).astype(str).unique())
            df[col] = pd.Categorical(df[col].fillna("missing").astype(str), categories=self.category_levels_.get(col)).codes.astype(np.float32)
            df[f"{col}_freq"] = _frequency_encode(df[col].astype(str))

        # A compact group-aggregation block resembling the paper's feature-processing stage.
        agg_groups = [c for c in ["card1", "card2", "addr1"] if c in df.columns]
        agg_cols = [c for c in ["TransactionAmt", "dist1", "dist2"] if c in df.columns]
        if agg_groups and agg_cols:
            for group_col in agg_groups:
                grouped = df.groupby(group_col, dropna=False)[agg_cols].agg(["mean", "std", "count"])
                grouped.columns = [f"{group_col}__{col[0]}_{col[1]}" for col in grouped.columns]
                df = df.merge(grouped, left_on=group_col, right_index=True, how="left")

        v_cols = _v_columns(df)
        if v_cols:
            v_values = df[v_cols].fillna(0).astype(np.float32)
            scaler = StandardScaler()
            v_scaled = scaler.fit_transform(v_values)
            pca = PCA(n_components=min(self.pca_components, v_scaled.shape[1]), random_state=RANDOM_STATE)
            comps = pca.fit_transform(v_scaled)
            for idx in range(comps.shape[1]):
                df[f"V_pca_{idx}"] = comps[:, idx].astype(np.float32)

        df["missing_count"] = df.isna().sum(axis=1).astype(np.float32)
        return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing").astype(str)
    return df


def _train_base_models(X_train: pd.DataFrame, y_train: pd.Series):
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": None,
        "lightgbm": None,
        "catboost": None,
        "logistic_regression": LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_weight=1,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        )
    except Exception:
        pass

    try:
        import lightgbm as lgb

        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function="Logloss",
            eval_metric="AUC:pr",
            verbose=False,
            random_seed=RANDOM_STATE,
        )
    except Exception:
        pass

    fitted = {}
    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def _model_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _weighted_vote(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    aligned = []
    total = 0.0
    for name, pred in predictions.items():
        weight = max(weights.get(name, 0.0), 0.0)
        if weight <= 0:
            continue
        aligned.append(pred * weight)
        total += weight
    if not aligned or total <= 0:
        return np.mean(list(predictions.values()), axis=0)
    return np.sum(aligned, axis=0) / total


def train_paper_style(data_path: str, sample: int | None = None, n_folds: int = 5):
    df = pd.read_csv(data_path)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")

    n = len(df)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    preprocessor = PaperPreprocessor()
    preprocessor.fit(_clean_dataframe(X_train), y_train)

    X_train_proc = preprocessor.transform(_clean_dataframe(X_train))
    X_val_proc = preprocessor.transform(_clean_dataframe(X_val))
    X_test_proc = preprocessor.transform(_clean_dataframe(X_test))

    base_models = _train_base_models(X_train_proc, y_train)

    val_predictions = {name: _model_predict_proba(model, X_val_proc) for name, model in base_models.items()}
    val_scores = {name: average_precision_score(y_val, pred) for name, pred in val_predictions.items()}
    weighted_vote_val = _weighted_vote(val_predictions, val_scores)

    # Stacking uses out-of-fold base predictions on the training split.
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_matrix = np.zeros((len(X_train_proc), len(base_models)))
    model_names = list(base_models.keys())

    for fold, (tr_idx, fold_idx) in enumerate(kf.split(X_train_proc)):
        fold_X_train = X_train_proc.iloc[tr_idx]
        fold_y_train = y_train.iloc[tr_idx]
        fold_X_valid = X_train_proc.iloc[fold_idx]
        fold_models = _train_base_models(fold_X_train, fold_y_train)
        for col_idx, name in enumerate(model_names):
            if name in fold_models:
                oof_matrix[fold_idx, col_idx] = _model_predict_proba(fold_models[name], fold_X_valid)
            else:
                oof_matrix[fold_idx, col_idx] = fold_y_train.mean()
        fold_pred = oof_matrix[fold_idx].mean(axis=1)
        print(f"fold {fold} pr-auc {average_precision_score(y_train.iloc[fold_idx], fold_pred):.6f}")

    meta = LogisticRegression(max_iter=1000, solver="liblinear")
    meta.fit(oof_matrix, y_train)

    test_matrix = np.column_stack([_model_predict_proba(model, X_test_proc) for model in base_models.values()])
    stacked_test = meta.predict_proba(test_matrix)[:, 1]
    weighted_vote_test = _weighted_vote({name: _model_predict_proba(model, X_test_proc) for name, model in base_models.items()}, val_scores)

    results = {
        "base_models": base_models,
        "preprocessor": preprocessor,
        "meta": meta,
        "val_scores": val_scores,
        "weighted_vote_val_pr_auc": average_precision_score(y_val, weighted_vote_val),
        "stacked_val_pr_auc": average_precision_score(y_val, meta.predict_proba(np.column_stack([_model_predict_proba(model, X_val_proc) for model in base_models.values()]))[:, 1]),
        "test_metrics": {
            "stacked_pr_auc": average_precision_score(y_test, stacked_test),
            "stacked_roc_auc": roc_auc_score(y_test, stacked_test),
            "weighted_vote_pr_auc": average_precision_score(y_test, weighted_vote_test),
            "weighted_vote_roc_auc": roc_auc_score(y_test, weighted_vote_test),
        },
        "oof_pr_auc": average_precision_score(y_train, oof_matrix.mean(axis=1)),
        "feature_count": X_train_proc.shape[1],
    }
    return results
