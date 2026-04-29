from typing import Iterator, Tuple
import pandas as pd


def time_based_splits(df: pd.DataFrame, time_col: str = "TransactionDT", train_frac: float = 0.7, val_frac: float = 0.15) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) temporal splits. Keeps order and avoids leakage.

    Simple generator: single split or sliding window could be added.
    """
    if time_col not in df.columns:
        raise ValueError("time_col not in df")
    df_sorted = df.sort_values(time_col).reset_index(drop=False)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_idx = df_sorted.iloc[:train_end]["index"].values
    val_idx = df_sorted.iloc[train_end:val_end]["index"].values
    yield (pd.Index(train_idx), pd.Index(val_idx))
