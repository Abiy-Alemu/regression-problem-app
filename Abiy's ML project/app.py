"""
Streamlit application for the Ethiopia socio-economic ML project.

Features:
- Load pre-trained regression, classification, and clustering models
- Allow users to:
  * Select a historical year (from the Ethiopia WDI data)
  * Optionally upload a custom CSV row with socio-economic indicators
  * Interactively adjust key indicators
- Produce regression predictions and growth regime classification
- Visualize:
  * Feature importance (Random Forest regression)
  * Clusters of years in 2D PCA space
  * Prediction trends over time

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from eth_ml.data import load_and_prepare_dataset
from eth_ml.utils import (
    ROOT_DIR,
    get_models_dir,
    load_joblib,
    load_json,
)


sns.set(style="whitegrid")


@st.cache_resource
def load_data_and_models() -> Tuple[
    pd.DataFrame,
    Dict[str, str],
    Any,
    Any,
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float],
    Dict[str, Any],
]:
    """Load dataset, trained models, metrics, and metadata (cached)."""
    features_df, indicator_meta = load_and_prepare_dataset()

    models_dir = get_models_dir()
    reg_model = load_joblib(models_dir / "regression_model.pkl")
    cls_model = load_joblib(models_dir / "classification_model.pkl")
    clustering_model = load_joblib(models_dir / "clustering_model.pkl")

    reg_metrics = load_json(models_dir / "regression_metrics.json")
    cls_metrics = load_json(models_dir / "classification_metrics.json")
    training_metadata = load_json(models_dir / "training_metadata.json")

    return (
        features_df,
        indicator_meta,
        reg_model,
        cls_model,
        clustering_model,
        reg_metrics,
        cls_metrics,
        training_metadata,
    )


def get_top_features_from_reg_model(
    reg_model, feature_names: list[str], top_n: int = 10
) -> pd.DataFrame:
    """Extract top-N feature importances from the Random Forest regressor."""
    rf = reg_model.named_steps.get("model", None)
    if rf is None or not hasattr(rf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = rf.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
    return df_imp


def build_input_vector_from_year(
    features_df: pd.DataFrame, year: int, feature_names: list[str]
) -> np.ndarray:
    """Return feature vector for a given year from the historical dataset."""
    row = features_df.loc[year, feature_names]
    return row.values.reshape(1, -1)


def build_input_vector_from_manual_inputs(
    features_df: pd.DataFrame,
    top_features: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Collect manual inputs from the user for a subset of features.
    Other features fall back to dataset medians.
    """
    medians = features_df.median()
    feature_names = features_df.columns.tolist()

    values: Dict[str, float] = {}

    st.write(
        "Adjust key indicators below. Defaults are based on the historical median."
    )

    for _, row in top_features.iterrows():
        f = row["feature"]
        series = features_df[f].dropna()
        if series.empty:
            continue
        min_val = float(series.quantile(0.05))
        max_val = float(series.quantile(0.95))
        default = float(medians[f])

        val = st.number_input(
            f,
            value=default,
            min_value=min_val,
            max_value=max_val,
            help="Socio-economic indicator value (approximate range from Ethiopia data).",
        )
        values[f] = float(val)

    # For non-top features, use medians
    for f in feature_names:
        if f not in values:
            values[f] = float(medians[f])

    x_vec = np.array([values[f] for f in feature_names], dtype=float).reshape(1, -1)
    return x_vec, values


def load_uploaded_row(
    uploaded_file, expected_features: list[str]
) -> Tuple[np.ndarray | None, str | None]:
    """
    Read a CSV uploaded by the user and extract a single row of features.
    The CSV must contain at least the expected feature columns.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read CSV: {e}"

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        return (
            None,
            "Uploaded CSV is missing required columns: " + ", ".join(missing),
        )

    if df.empty:
        return None, "Uploaded CSV is empty."

    row = df.iloc[0][expected_features].values.reshape(1, -1)
    return row, None


def plot_feature_importance(df_imp: pd.DataFrame) -> None:
    """Plot feature importance bar chart."""
    if df_imp.empty:
        st.info("Feature importance is not available for this model.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_imp, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title("Top feature importances (Random Forest Regressor)")
    st.pyplot(fig)


def plot_prediction_trend(
    features_df: pd.DataFrame,
    reg_model,
    target_code: str,
    indicator_meta: Dict[str, str],
) -> None:
    """Plot actual vs. predicted regression target over years."""
    df = features_df.dropna(subset=[target_code])
    X = df.drop(columns=[target_code]).values
    y_true = df[target_code].values
    years = df.index.values

    y_pred = reg_model.predict(X)

    target_name = indicator_meta.get(target_code, target_code)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y_true, label="Actual", marker="o")
    ax.plot(years, y_pred, label="Predicted", marker="s")
    ax.set_xlabel("Year")
    ax.set_ylabel(target_name)
    ax.set_title("Actual vs. predicted target over time")
    ax.legend()
    st.pyplot(fig)


def plot_clusters(
    clustering_model: Dict[str, Any],
    highlighted_point: np.ndarray | None = None,
) -> None:
    """
    Visualize KMeans clusters in PCA space and optionally highlight a point.
    """
    X_pca = np.array(clustering_model["X_pca"])
    cluster_labels = np.array(clustering_model["cluster_labels"])
    years = np.array(clustering_model["years"])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=80,
        edgecolor="k",
    )
    for i, year in enumerate(years):
        ax.text(
            X_pca[i, 0],
            X_pca[i, 1],
            str(year),
            fontsize=8,
            ha="center",
            va="center",
        )

    if highlighted_point is not None:
        ax.scatter(
            highlighted_point[0],
            highlighted_point[1],
            c="red",
            s=120,
            marker="X",
            label="Current input",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("KMeans clusters of socio-economic years (PCA)")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(
        page_title="Ethiopia Socio-Economic ML",
        layout="wide",
    )

    st.title("Ethiopia Socio-Economic Machine Learning App")
    st.markdown(
        """
This interactive app demonstrates regression, classification, and clustering
models built on Ethiopia's World Bank WDI socio-economic indicators.

- **Left sidebar**: choose input mode and adjust indicators.
- **Main area**: see predictions, model performance, and visualizations.
"""
    )

    models_dir = get_models_dir()
    missing_files = [
        models_dir / "regression_model.pkl",
        models_dir / "classification_model.pkl",
        models_dir / "clustering_model.pkl",
        models_dir / "training_metadata.json",
    ]
    if any(not p.exists() for p in missing_files):
        st.error(
            "Trained models are missing. Please run `python train_models.py` "
            "once before launching the app."
        )
        st.stop()

    (
        features_df,
        indicator_meta,
        reg_model,
        cls_model,
        clustering_model,
        reg_metrics,
        cls_metrics,
        training_metadata,
    ) = load_data_and_models()

    target_code = training_metadata["target_code"]
    target_name = training_metadata.get("target_name", target_code)
    reg_feature_names = training_metadata["regression_feature_names"]
    cls_feature_names = training_metadata["classification_feature_names"]
    class_names = training_metadata["class_names"]

    # Sidebar: input mode
    st.sidebar.header("Input configuration")
    input_mode = st.sidebar.radio(
        "Choose input mode:",
        options=["Select historical year", "Upload CSV row", "Manual key indicators"],
    )

    x_reg = None
    manual_values = None

    if input_mode == "Select historical year":
        year = st.sidebar.selectbox(
            "Select year from Ethiopia WDI data",
            options=sorted(features_df.index.tolist()),
        )
        x_reg = build_input_vector_from_year(features_df[reg_feature_names], year, reg_feature_names)
        st.sidebar.info(f"Using historical indicators for year {year}.")

    elif input_mode == "Upload CSV row":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV with indicator columns (first row will be used)",
            type=["csv"],
        )
        if uploaded is not None:
            x_reg, err = load_uploaded_row(uploaded, reg_feature_names)
            if err:
                st.sidebar.error(err)
                x_reg = None
        else:
            st.sidebar.warning("Upload a CSV file to enable this mode.")

    else:  # Manual key indicators
        df_imp = get_top_features_from_reg_model(reg_model, reg_feature_names, top_n=8)
        x_reg, manual_values = build_input_vector_from_manual_inputs(
            features_df[reg_feature_names], df_imp
        )

    st.sidebar.markdown("---")
    run_prediction = st.sidebar.button("Run prediction", disabled=x_reg is None)

    col_pred, col_perf = st.columns([2, 1])

    with col_pred:
        st.subheader("Regression & classification outputs")
        if x_reg is None:
            st.info("Provide valid inputs in the sidebar and click **Run prediction**.")
        elif run_prediction:
            try:
                # Regression prediction
                y_reg = float(reg_model.predict(x_reg)[0])
                st.metric(
                    label=f"Predicted target ({target_name})",
                    value=f"{y_reg:,.2f}",
                )

                # Classification prediction (growth regime)
                # Use the overlapping feature set as a best-effort approximation
                x_cls_df = pd.DataFrame(x_reg, columns=reg_feature_names)
                x_cls_df = x_cls_df.reindex(columns=cls_feature_names, axis=1)
                x_cls = x_cls_df.values

                y_cls = int(cls_model.predict(x_cls)[0])
                label = class_names[y_cls]

                if hasattr(cls_model, "predict_proba"):
                    probs = cls_model.predict_proba(x_cls)[0]
                    prob_str = ", ".join(
                        f"{name}: {p:.2f}" for name, p in zip(class_names, probs)
                    )
                else:
                    prob_str = "Probabilities not available for this model."

                st.write(f"**Predicted growth regime:** {label}")
                st.caption(prob_str)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

    with col_perf:
        st.subheader("Model performance summary")
        st.markdown("**Regression (RandomForestRegressor)**")
        st.json(reg_metrics)
        st.markdown("**Classification (RandomForestClassifier)**")
        st.json(cls_metrics)

    st.markdown("---")
    col_feat, col_trend = st.columns(2)

    with col_feat:
        st.subheader("Feature importance (regression)")
        df_imp = get_top_features_from_reg_model(reg_model, reg_feature_names)
        plot_feature_importance(df_imp)

    with col_trend:
        st.subheader("Prediction trends over time")
        plot_prediction_trend(features_df, reg_model, target_code, indicator_meta)

    st.markdown("---")
    st.subheader("Clustering of socio-economic years")

    # Compute PCA coordinates for current input if available
    highlighted_point = None
    if x_reg is not None:
        try:
            scaler = clustering_model["scaler"]
            pca = clustering_model["pca"]
            x_scaled = scaler.transform(x_reg)
            highlighted_point = pca.transform(x_scaled)[0]
        except Exception:
            highlighted_point = None

    plot_clusters(clustering_model, highlighted_point=highlighted_point)

    st.markdown("---")
    st.markdown(
        f"Models directory: `{get_models_dir().relative_to(ROOT_DIR)}` – "
        "contains the persisted pipelines and metadata used by this app."
    )


if __name__ == "__main__":
    main()

