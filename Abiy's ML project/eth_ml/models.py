"""
Model construction utilities for the Ethiopia socio-economic ML project.

This module defines sklearn Pipelines and objects used both during training
and at inference time in the Streamlit app.
"""

from __future__ import annotations

from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def build_regression_pipeline() -> Pipeline:
    """
    Regression pipeline:
    - Median imputation
    - Standard scaling
    - RandomForestRegressor (ensemble, non-linear)
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_classification_pipeline() -> Pipeline:
    """
    Classification pipeline:
    - Median imputation
    - Standard scaling
    - RandomForestClassifier (ensemble, non-linear)
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_clustering_objects(
    n_clusters: int = 3,
) -> Tuple[StandardScaler, KMeans, PCA]:
    """
    Objects used for unsupervised clustering:
    - StandardScaler
    - KMeans
    - PCA (2D) for visualization
    """
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pca = PCA(n_components=2, random_state=42)
    return scaler, kmeans, pca

