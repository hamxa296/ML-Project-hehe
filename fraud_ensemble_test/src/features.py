from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


def _as_frame(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def _stringify(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("missing").str.strip().str.lower()


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_flag_threshold: float = 0.20):
        self.missing_flag_threshold = missing_flag_threshold
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.high_missing_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.category_frequencies_: dict[str, dict[str, float]] = {}
        self.category_modes_: dict[str, str] = {}
        self.low_cardinality_categories_: dict[str, list[str]] = {}
        self.group_maps_: dict[str, dict[str, float]] = {}
        self.group_amount_medians_: dict[str, dict[str, float]] = {}
        self.amount_median_: float = 0.0
        self.amount_mean_: float = 0.0
        self.amount_std_: float = 1.0
        self.amount_clip_high_: float = 0.0
        self.txdt_median_: float = 0.0
        self.text_columns_: list[str] = []
        self.engineered_columns_: list[str] = []

    def fit(self, X, y=None):
        X_df = _as_frame(X)
        numeric = X_df.select_dtypes(include=[np.number])
        categorical = X_df.select_dtypes(include=["object", "string", "category"])

        self.numeric_columns_ = list(numeric.columns)
        self.categorical_columns_ = list(categorical.columns)
        self.text_columns_ = [
            col
            for col in self.categorical_columns_
            if col in {"DeviceInfo", "id_30", "id_31", "P_emaildomain", "R_emaildomain"}
        ]

        self.high_missing_columns_ = [
            col for col in X_df.columns if X_df[col].isna().mean() >= self.missing_flag_threshold
        ]
        self.numeric_medians_ = numeric.median(numeric_only=True).fillna(0.0).to_dict()

        for col in self.categorical_columns_:
            clean = _stringify(X_df[col])
            self.category_frequencies_[col] = clean.value_counts(normalize=True, dropna=False).to_dict()
            mode_value = clean.mode(dropna=False)
            self.category_modes_[col] = mode_value.iloc[0] if not mode_value.empty else "missing"
            unique_values = clean.nunique(dropna=True)
            if unique_values <= 12:
                self.low_cardinality_categories_[col] = sorted(clean.dropna().unique().tolist())

        if "TransactionAmt" in X_df.columns:
            amt = pd.to_numeric(X_df["TransactionAmt"], errors="coerce")
            self.amount_median_ = float(amt.median()) if amt.notna().any() else 0.0
            self.amount_mean_ = float(amt.mean()) if amt.notna().any() else 0.0
            self.amount_std_ = float(amt.std(ddof=0)) if amt.notna().any() else 1.0
            if not np.isfinite(self.amount_std_) or self.amount_std_ == 0:
                self.amount_std_ = 1.0
            self.amount_clip_high_ = float(amt.quantile(0.995)) if amt.notna().any() else 0.0
        if "TransactionDT" in X_df.columns:
            txdt = pd.to_numeric(X_df["TransactionDT"], errors="coerce")
            self.txdt_median_ = float(txdt.median()) if txdt.notna().any() else 0.0

        group_specs = [
            ["card1"],
            ["card1", "addr1"],
            ["card1", "card2"],
            ["card1", "card3"],
            ["card1", "card5"],
            ["card1", "card6"],
            ["card1", "P_emaildomain"],
            ["card1", "R_emaildomain"],
            ["card1", "DeviceType"],
            ["card1", "DeviceInfo"],
            ["card1", "id_30"],
            ["card1", "id_31"],
            ["card1", "addr1", "P_emaildomain"],
        ]

        amount_series = None
        if "TransactionAmt" in X_df.columns:
            amount_series = pd.to_numeric(X_df["TransactionAmt"], errors="coerce").fillna(self.amount_median_)

        for cols in group_specs:
            if not all(col in X_df.columns for col in cols):
                continue
            group_key = "__".join(cols)
            key_series = self._build_key(X_df, cols)
            self.group_maps_[group_key] = key_series.value_counts(dropna=False).to_dict()
            if amount_series is not None:
                self.group_amount_medians_[group_key] = amount_series.groupby(key_series).median().to_dict()

        self.engineered_columns_ = self._build_feature_frame(X_df).columns.tolist()
        return self

    def _build_key(self, X_df: pd.DataFrame, cols: list[str]) -> pd.Series:
        key = pd.Series("", index=X_df.index, dtype="string")
        for col in cols:
            if col in X_df.columns:
                key = key + "|" + _stringify(X_df[col])
            else:
                key = key + "|missing"
        return key.str.lstrip("|")

    def _build_feature_frame(self, X_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=X_df.index)
        feature_dict = {}

        if "TransactionDT" in X_df.columns:
            txdt = pd.to_numeric(X_df["TransactionDT"], errors="coerce").fillna(self.txdt_median_)
        else:
            txdt = pd.Series(self.txdt_median_, index=X_df.index, dtype=float)
        feature_dict["TransactionDT"] = txdt
        feature_dict["TransactionDT_log1p"] = np.log1p(txdt.clip(lower=0))
        tx_hour = ((txdt // 3600) % 24).astype(float)
        tx_day = ((txdt // 86400) % 7).astype(float)
        feature_dict["tx_hour"] = tx_hour
        feature_dict["tx_day"] = tx_day
        feature_dict["tx_hour_sin"] = np.sin(2 * np.pi * tx_hour / 24.0)
        feature_dict["tx_hour_cos"] = np.cos(2 * np.pi * tx_hour / 24.0)
        feature_dict["tx_day_sin"] = np.sin(2 * np.pi * tx_day / 7.0)
        feature_dict["tx_day_cos"] = np.cos(2 * np.pi * tx_day / 7.0)

        if "TransactionAmt" in X_df.columns:
            amt = pd.to_numeric(X_df["TransactionAmt"], errors="coerce").fillna(self.amount_median_)
        else:
            amt = pd.Series(self.amount_median_, index=X_df.index, dtype=float)
        amt = amt.clip(lower=0, upper=self.amount_clip_high_ if self.amount_clip_high_ > 0 else None)
        feature_dict["TransactionAmt"] = amt
        feature_dict["TransactionAmt_log1p"] = np.log1p(amt)
        feature_dict["TransactionAmt_sqrt"] = np.sqrt(amt)
        feature_dict["TransactionAmt_zscore"] = (amt - self.amount_mean_) / (self.amount_std_ + 1e-6)

        numeric_frame = X_df.select_dtypes(include=[np.number])
        for col in self.numeric_columns_:
            series = pd.to_numeric(X_df[col], errors="coerce") if col in X_df.columns else pd.Series(np.nan, index=X_df.index)
            filled = series.fillna(self.numeric_medians_.get(col, 0.0))
            feature_dict[col] = filled
            if col in self.high_missing_columns_:
                feature_dict[f"{col}_missing"] = series.isna().astype(np.int8)

        for col in self.categorical_columns_:
            clean = _stringify(X_df[col]) if col in X_df.columns else pd.Series("missing", index=X_df.index)
            freq_map = self.category_frequencies_.get(col, {})
            feature_dict[f"{col}_freq"] = clean.map(freq_map).fillna(0.0).astype(float)
            feature_dict[f"{col}_len"] = clean.str.len().fillna(0).astype(float)
            feature_dict[f"{col}_missing"] = (clean == "missing").astype(np.int8)

            if col in self.low_cardinality_categories_:
                categories = self.low_cardinality_categories_[col]
                for category in categories:
                    feature_dict[f"{col}__{category}"] = (clean == category).astype(np.int8)

            if col in self.text_columns_:
                lowered = clean
                for token in ["windows", "mac", "ios", "android", "linux", "chrome", "safari", "firefox", "mobile", "tablet", "samsung"]:
                    feature_dict[f"{col}__has_{token}"] = lowered.str.contains(token, regex=False, na=False).astype(np.int8)

        features = pd.DataFrame(feature_dict)
        if "TransactionAmt" in feature_dict:
            amt_feature = feature_dict["TransactionAmt"]
            for group_key, mapping in self.group_maps_.items():
                cols = group_key.split("__")
                key_series = self._build_key(X_df, cols)
                count_feature = key_series.map(mapping).fillna(0.0).astype(float)
                feature_dict[f"{group_key}__count"] = count_feature
                amt_medians = self.group_amount_medians_.get(group_key, {})
                median_feature = key_series.map(amt_medians).fillna(self.amount_median_).astype(float)
                feature_dict[f"{group_key}__amt_median"] = median_feature
                feature_dict[f"{group_key}__amt_ratio"] = amt_feature / (median_feature + 1e-6)

            features = pd.DataFrame(feature_dict)

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features

    def _add_velocity_features(self, X_df: pd.DataFrame, feature_dict: dict, windows_seconds: list[int] | None = None) -> None:
        if windows_seconds is None:
            windows_seconds = [3600, 6 * 3600, 24 * 3600]
        if "TransactionDT" not in X_df.columns:
            return
        tx = pd.to_numeric(X_df["TransactionDT"], errors="coerce").fillna(self.txdt_median_)
        for entity in ["card1", "DeviceInfo", "addr1", "P_emaildomain"]:
            if entity not in X_df.columns:
                continue
            key = _stringify(X_df[entity])
            df_tmp = pd.DataFrame({"tx": tx, "key": key})
            order = df_tmp.sort_values("tx").reset_index()
            for w in windows_seconds:
                col_name = f"vel__{entity}__cnt_{w}s"
                counts = np.zeros(len(X_df), dtype=float)
                groups = defaultdict(list)
                for idx, row in order.iterrows():
                    k = row["key"]
                    t = float(row["tx"])
                    groups[k].append((t, int(row["index"])))
                for k, recs in groups.items():
                    times = [r[0] for r in recs]
                    idxs = [r[1] for r in recs]
                    L = 0
                    for R in range(len(times)):
                        while times[R] - times[L] > w:
                            L += 1
                        counts[idxs[R]] = R - L
                feature_dict[col_name] = counts

    def _add_entity_graph_features(self, X_df: pd.DataFrame, feature_dict: dict) -> None:
        mappings = [
            ("card1", "P_emaildomain"),
            ("card1", "DeviceInfo"),
            ("card1", "addr1"),
            ("P_emaildomain", "card1"),
        ]
        for a, b in mappings:
            if a not in X_df.columns or b not in X_df.columns:
                continue
            a_key = _stringify(X_df[a])
            b_key = _stringify(X_df[b])
            df_pair = pd.DataFrame({"a": a_key, "b": b_key})
            uniq = df_pair.groupby("a")["b"].nunique()
            feature_dict[f"entity__{a}__unique_{b}"] = a_key.map(uniq).fillna(0).astype(float)
            uniq2 = df_pair.groupby("b")["a"].nunique()
            feature_dict[f"entity__{b}__unique_{a}"] = b_key.map(uniq2).fillna(0).astype(float)

    def transform(self, X):
        X_df = _as_frame(X)
        features = self._build_feature_frame(X_df)

        # augment with velocity and entity features on-the-fly (they are deterministic)
        try:
            self._add_velocity_features(X_df, features)
        except Exception:
            pass
        try:
            self._add_entity_graph_features(X_df, features)
        except Exception:
            pass

        for col in self.engineered_columns_:
            if col not in features.columns:
                features[col] = 0.0

        return features.reindex(columns=self.engineered_columns_, fill_value=0.0)


class ClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.features_to_cluster = None

    def fit(self, X, y=None):
        X_df = _as_frame(X)
        self.features_to_cluster = list(X_df.columns)
        self.kmeans.fit(X_df.fillna(0.0))
        return self

    def transform(self, X):
        X_df = _as_frame(X)
        clusters = self.kmeans.predict(X_df[self.features_to_cluster].fillna(0.0))
        X_out = X_df.copy()
        X_out["cluster_label"] = clusters
        return X_out
