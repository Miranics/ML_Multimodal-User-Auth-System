from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MERGED_FEATURES_PATH, SOCIAL_PROFILES_PATH, TRANSACTIONS_PATH
from src.utils import infer_common_id_column


def load_and_merge(
    social_path: Path = SOCIAL_PROFILES_PATH,
    tx_path: Path = TRANSACTIONS_PATH,
) -> pd.DataFrame:
    if not social_path.exists() or not tx_path.exists():
        raise FileNotFoundError(
            "Missing raw tabular files. Expected: "
            f"{social_path} and {tx_path}."
        )

    social_df = pd.read_csv(social_path)
    tx_df = pd.read_csv(tx_path)

    key_col = infer_common_id_column(social_df.columns.tolist(), tx_df.columns.tolist())

    merged = social_df.merge(tx_df, on=key_col, how="inner", suffixes=("_social", "_tx"))
    if merged.empty:
        raise ValueError("Merged table is empty. Verify matching IDs in both datasets.")

    merged = engineer_features(merged)
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Datetime decomposition
    for col in out.columns:
        col_l = col.lower()
        if "date" in col_l or "time" in col_l:
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().sum() > 0:
                out[f"{col}_year"] = parsed.dt.year
                out[f"{col}_month"] = parsed.dt.month
                out[f"{col}_day"] = parsed.dt.day
                out[f"{col}_dow"] = parsed.dt.dayofweek

    # Basic numeric imputation
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = out[col].fillna(out[col].median())

    # Low-cardinality categorical cleanup
    cat_cols = out.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        out[col] = out[col].astype(str).str.strip().replace({"": "unknown"})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge customer profile and transaction data.")
    parser.add_argument("--social", type=Path, default=SOCIAL_PROFILES_PATH)
    parser.add_argument("--transactions", type=Path, default=TRANSACTIONS_PATH)
    parser.add_argument("--out", type=Path, default=MERGED_FEATURES_PATH)
    args = parser.parse_args()

    merged = load_and_merge(args.social, args.transactions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"Merged dataset shape: {merged.shape}")
    print(f"Saved merged dataset to: {args.out}")


if __name__ == "__main__":
    main()
