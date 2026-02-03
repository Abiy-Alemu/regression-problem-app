"""
Data loading and preprocessing utilities for the Ethiopia socio-economic ML project.

This module is intentionally self-contained so it can be reused by:
- the offline training script (`train_models.py`)
- the interactive Streamlit application (`app.py`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Project root is the parent of this `eth_ml` package directory
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "API_ETH_DS2_en_csv_v2_6515.csv"


def get_default_data_path() -> Path:
    """Return the default path to the Ethiopia WDI CSV."""
    return DEFAULT_DATA_PATH


def load_raw_world_bank_data(path: Path | str) -> pd.DataFrame:
    """
    Load the raw World Bank Ethiopia dataset.

    The file starts with metadata rows. For World Bank WDI exports, the
    first 4 rows are metadata; the 5th row is the actual header.
    We therefore use `skiprows=4`.
    """
    path = Path(path)
    df = pd.read_csv(path, skiprows=4)
    return df


def filter_ethiopia(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows for Ethiopia."""
    eth = df[df["Country Name"] == "Ethiopia"].copy()
    return eth


def long_to_year_indicator_matrix(
    df_eth: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert Ethiopia indicator table from wide format (years as columns)
    to a Year × IndicatorCode numeric matrix.

    Returns
    -------
    data_year_indicator : pd.DataFrame
        Index: Year (int), Columns: Indicator Code, Values: indicator values.
    indicator_meta : dict
        Mapping `Indicator Code -> Indicator Name`.
    """
    non_year_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [c for c in df_eth.columns if c not in non_year_cols]

    df_long = df_eth.melt(
        id_vars=non_year_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Year"])
    df_long["Year"] = df_long["Year"].astype(int)

    indicator_meta = (
        df_long[["Indicator Code", "Indicator Name"]]
        .drop_duplicates()
        .set_index("Indicator Code")["Indicator Name"]
        .to_dict()
    )

    data_year_indicator = df_long.pivot_table(
        index="Year", columns="Indicator Code", values="Value"
    ).sort_index()

    return data_year_indicator, indicator_meta


def select_numeric_features(
    data_year_indicator: pd.DataFrame, min_non_null_ratio: float = 0.7
) -> pd.DataFrame:
    """
    Select numeric indicator columns with at least `min_non_null_ratio`
    non-missing values across years.
    """
    numeric_df = data_year_indicator.select_dtypes(include=[np.number])
    non_null_ratio = numeric_df.notna().mean(axis=0)
    valid_cols = non_null_ratio[non_null_ratio >= min_non_null_ratio].index
    reduced_df = numeric_df[valid_cols].copy()
    return reduced_df


def load_and_prepare_dataset(
    path: Path | str | None = None, min_non_null_ratio: float = 0.7
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    End-to-end data loading + basic feature selection:

    1. Load raw WDI data
    2. Filter Ethiopia
    3. Convert to Year × Indicator matrix
    4. Keep numeric features with enough data
    """
    if path is None:
        path = DEFAULT_DATA_PATH
    df_raw = load_raw_world_bank_data(path)
    df_eth = filter_ethiopia(df_raw)
    data_year_indicator, indicator_meta = long_to_year_indicator_matrix(df_eth)
    features_df = select_numeric_features(data_year_indicator, min_non_null_ratio)
    return features_df, indicator_meta


def choose_regression_target(data: pd.DataFrame) -> str:
    """
    Choose a sensible regression target indicator code.

    Preference order (if present in columns):
    - Merchandise exports (current US$)
    - Export value index
    - GDP per capita
    - GDP growth
    Fallback: first column in the DataFrame.
    """
    preferred_order = [
        "TX.VAL.MRCH.CD.WT",  # Merchandise exports (current US$)
        "TX.VAL.MRCH.XD.WD",  # Export value index (2015 = 100)
        "NY.GDP.PCAP.KD",  # GDP per capita (constant 2015 US$)
        "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    ]
    for code in preferred_order:
        if code in data.columns:
            return code
    return data.columns[0]


def create_growth_categories(
    data: pd.DataFrame, target_code: str
) -> pd.DataFrame:
    """
    Create a categorical growth target from a continuous series.

    Steps:
    - Take the chosen continuous target over time.
    - Compute percentage change (year-over-year growth).
    - Bin growth into 3 categories:
        * Low/Negative (<= -5%)
        * Stable (-5% to 5%)
        * High (> 5%)
    - If bins collapse (e.g., too few classes), fall back to quantile-based bins.

    Returns
    -------
    df_cls : pd.DataFrame
        DataFrame aligned on years where both features and class labels exist.
        Contains:
        - All numeric indicator features
        - `growth_rate` (continuous)
        - `growth_category` (categorical target)
    """
    series = data[target_code].dropna()
    growth_rate = series.pct_change()

    df_tmp = pd.DataFrame(
        {
            "Year": series.index,
            "target_value": series.values,
            "growth_rate": growth_rate.values,
        }
    ).set_index("Year")

    df_tmp = df_tmp.dropna(subset=["growth_rate"])

    df_tmp["growth_category"] = pd.cut(
        df_tmp["growth_rate"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["Low/Negative", "Stable", "High"],
    )

    if df_tmp["growth_category"].nunique() < 3:
        df_tmp["growth_category"] = pd.qcut(
            df_tmp["growth_rate"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

    df_features = data.loc[df_tmp.index].copy()
    df_cls = df_features.copy()
    df_cls["growth_rate"] = df_tmp["growth_rate"]
    df_cls["growth_category"] = df_tmp["growth_category"]

    df_cls = df_cls.dropna(subset=["growth_category"])
    return df_cls

