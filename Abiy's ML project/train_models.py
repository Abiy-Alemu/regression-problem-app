"""
Offline training script for the Ethiopia socio-economic ML project.

This script:
- Loads and preprocesses the Ethiopia WDI dataset
- Trains:
  * RandomForestRegressor for the regression task
  * RandomForestClassifier for the classification task
  * StandardScaler + KMeans + PCA for clustering
- Evaluates models and saves key metrics
- Persists trained models and metadata as .pkl / .json files

Models and metadata are saved under the `models/` directory so that the
Streamlit application (`app.py`) can load them for inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from eth_ml.data import (
    load_and_prepare_dataset,
    choose_regression_target,
    create_growth_categories,
)
from eth_ml.models import (
    build_regression_pipeline,
    build_classification_pipeline,
    build_clustering_objects,
)
from eth_ml.utils import get_models_dir, save_joblib, save_json


def train_regression(
    features_df: pd.DataFrame, target_code: str
) -> Dict[str, Any]:
    """Train and evaluate the regression model."""
    df_reg = features_df.dropna(subset=[target_code])
    X = df_reg.drop(columns=[target_code]).values
    y = df_reg[target_code].values
    feature_names = df_reg.drop(columns=[target_code]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg_pipeline = build_regression_pipeline()
    reg_pipeline.fit(X_train, y_train)

    y_pred_test = reg_pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

    return {
        "model": reg_pipeline,
        "metrics": metrics,
        "feature_names": feature_names,
    }


def train_classification(
    features_df: pd.DataFrame, target_code: str
) -> Dict[str, Any]:
    """Train and evaluate the classification model."""
    df_cls = create_growth_categories(features_df, target_code)

    y_cat = df_cls["growth_category"].astype("category")
    X = df_cls.drop(columns=["growth_category", "growth_rate"])

    feature_names = X.columns.tolist()
    X = X.values
    y = y_cat.cat.codes.values
    class_names = y_cat.cat.categories.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cls_pipeline = build_classification_pipeline()
    cls_pipeline.fit(X_train, y_train)

    y_pred_test = cls_pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred_test))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average="weighted", zero_division=0
    )

    metrics = {
        "Accuracy": float(acc),
        "Precision(weighted)": float(prec),
        "Recall(weighted)": float(rec),
        "F1(weighted)": float(f1),
    }

    return {
        "model": cls_pipeline,
        "metrics": metrics,
        "feature_names": feature_names,
        "class_names": class_names,
    }


def train_clustering(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Fit scaler, KMeans, and PCA for clustering and visualization."""
    scaler, kmeans, pca = build_clustering_objects(n_clusters=3)
    X = features_df.values

    X_scaled = scaler.fit_transform(X)
    clusters = kmeans.fit_predict(X_scaled)
    X_pca = pca.fit_transform(X_scaled)

    years = features_df.index.to_list()
    feature_names = features_df.columns.to_list()

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "feature_names": feature_names,
        "years": years,
        "cluster_labels": clusters.tolist(),
        "X_pca": X_pca.tolist(),
    }


def main() -> None:
    """Train all models and save them with their metadata."""
    features_df, indicator_meta = load_and_prepare_dataset()
    target_code = choose_regression_target(features_df)
    target_name = indicator_meta.get(target_code, target_code)

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Regression
    print("Training regression model...")
    reg_info = train_regression(features_df, target_code)
    save_joblib(reg_info["model"], models_dir / "regression_model.pkl")
    save_json(reg_info["metrics"], models_dir / "regression_metrics.json")

    # Classification
    print("Training classification model...")
    cls_info = train_classification(features_df, target_code)
    save_joblib(cls_info["model"], models_dir / "classification_model.pkl")
    save_json(cls_info["metrics"], models_dir / "classification_metrics.json")

    # Clustering
    print("Training clustering model...")
    clustering_info = train_clustering(features_df)
    save_joblib(clustering_info, models_dir / "clustering_model.pkl")

    # Shared metadata
    training_metadata = {
        "target_code": target_code,
        "target_name": target_name,
        "regression_feature_names": reg_info["feature_names"],
        "classification_feature_names": cls_info["feature_names"],
        "class_names": cls_info["class_names"],
        "indicator_meta": indicator_meta,
    }
    save_json(training_metadata, models_dir / "training_metadata.json")

    print("\n=== Training complete ===")
    print("Regression metrics:", reg_info["metrics"])
    print("Classification metrics:", cls_info["metrics"])
    print(f"Models and metadata saved under: {models_dir.resolve()}")


if __name__ == "__main__":
    main()

