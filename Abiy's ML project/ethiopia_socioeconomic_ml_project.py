"""
# Ethiopia Socio-Economic ML Project

End-to-end machine learning project using World Bank Ethiopia
(`API_ETH_DS2_en_csv_v2_6515.csv`) to illustrate:
- Data understanding & preprocessing
- Exploratory Data Analysis (EDA)
- Supervised learning (regression & classification)
- Unsupervised learning (clustering)
- Model comparison, bias–variance, and interpretation

This script is written in a notebook-friendly style with `# %%` cells and
markdown-style comments, so it can be run as a plain Python file or imported
into Jupyter / VS Code as a notebook.
"""

# %% [markdown]
"""
### 0. Imports and configuration

We import the standard data science stack:
- **pandas, numpy** for data manipulation
- **matplotlib, seaborn** for visualization
- **scikit-learn** for preprocessing, modeling, and evaluation
"""

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")


# %% [markdown]
"""
### 1. Data loading and understanding

The World Bank WDI CSV has the following structure:
- First a few **metadata rows** (source, last updated date)
- Then a **header row** containing:
  - `Country Name`, `Country Code`, `Indicator Name`, `Indicator Code`
  - One column per **year** (e.g. `1960`, `1961`, ..., `2024`)
- Each **row** corresponds to one **indicator** for one country.

For this project we:
1. Load the dataset
2. Filter for **Ethiopia**
3. Reshape to get a **Year × Indicator** matrix suitable for ML
"""

# %%
DATA_PATH = Path("API_ETH_DS2_en_csv_v2_6515.csv")


def load_raw_world_bank_data(path: Path) -> pd.DataFrame:
    """
    Load the raw World Bank Ethiopia dataset.

    The file starts with metadata rows. For World Bank WDI exports, the
    first 4 rows are metadata; the 5th row is the actual header.
    We therefore use `skiprows=4`.
    """
    df = pd.read_csv(path, skiprows=4)
    return df


def filter_ethiopia(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows for Ethiopia."""
    eth = df[df["Country Name"] == "Ethiopia"].copy()
    return eth


def long_to_year_indicator_matrix(df_eth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    # Identify the year columns (all columns that look like years)
    non_year_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [c for c in df_eth.columns if c not in non_year_cols]

    # Melt to long format: one row per (indicator, year)
    df_long = df_eth.melt(
        id_vars=non_year_cols, value_vars=year_cols, var_name="Year", value_name="Value"
    )

    # Clean and convert Year to integer, drop completely missing values
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long = df_long.dropna(subset=["Year"])
    df_long["Year"] = df_long["Year"].astype(int)

    # Create indicator metadata (code -> descriptive name)
    indicator_meta = (
        df_long[["Indicator Code", "Indicator Name"]]
        .drop_duplicates()
        .set_index("Indicator Code")["Indicator Name"]
        .to_dict()
    )

    # Pivot to Year × Indicator matrix
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
    path: Path, min_non_null_ratio: float = 0.7
) -> tuple[pd.DataFrame, dict]:
    """
    End-to-end data loading + basic feature selection:

    1. Load raw WDI data
    2. Filter Ethiopia
    3. Convert to Year × Indicator matrix
    4. Keep numeric features with enough data
    """
    df_raw = load_raw_world_bank_data(path)
    df_eth = filter_ethiopia(df_raw)
    data_year_indicator, indicator_meta = long_to_year_indicator_matrix(df_eth)
    features_df = select_numeric_features(data_year_indicator, min_non_null_ratio)
    return features_df, indicator_meta


# %% [markdown]
"""
### 2. Exploratory Data Analysis (EDA)

EDA steps:
- **Summary statistics**: central tendency and dispersion of indicators
- **Correlation heatmap**: relationships between indicators
- **Trend analysis over years**: visualize how selected indicators evolve
- **Key insights**: discussed in comments/markdown
"""

# %%
def perform_eda(
    data: pd.DataFrame, indicator_meta: dict, max_corr_features: int = 15
) -> None:
    """
    Basic EDA plots and statistics for the Year × Indicator matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Year × IndicatorCode matrix (numeric).
    indicator_meta : dict
        IndicatorCode -> IndicatorName mapping.
    max_corr_features : int
        Maximum number of features to include in the correlation heatmap
        (to keep the plot readable).
    """
    print("Data shape (years × indicators):", data.shape)
    print("\nFirst 5 rows (years as index, indicator codes as columns):")
    print(data.head())

    # Summary statistics
    print("\nSummary statistics (first 10 indicators):")
    display(data.describe().iloc[:, :10])

    # Correlation heatmap (subset of features with highest variance)
    variances = data.var().sort_values(ascending=False)
    top_features = variances.head(max_corr_features).index
    corr = data[top_features].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (top-variance socio-economic indicators)")
    plt.tight_layout()
    plt.show()

    # Trend plots for a few key indicators:
    # We attempt to use well-known socio-economic indicators if present; otherwise
    # we fall back to the first few high-variance indicators.
    preferred_codes = {
        "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
        "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
        "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
        "TX.VAL.MRCH.CD.WT": "Merchandise exports (current US$)",
    }
    available_pref = {c: n for c, n in preferred_codes.items() if c in data.columns}

    if available_pref:
        codes_to_plot = list(available_pref.keys())[:4]
        names_to_plot = [available_pref[c] for c in codes_to_plot]
    else:
        # Fallback: use first 4 high-variance indicators
        codes_to_plot = list(top_features[:4])
        names_to_plot = [indicator_meta.get(c, c) for c in codes_to_plot]

    plt.figure(figsize=(14, 8))
    for code, name in zip(codes_to_plot, names_to_plot):
        plt.plot(data.index, data[code], marker="o", label=name)
    plt.xlabel("Year")
    plt.ylabel("Indicator value")
    plt.title("Trends of selected socio-economic indicators over time (Ethiopia)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        "\nKey EDA observations (to be refined by the student when running the code):"
    )
    print(
        "- Look for monotonic upward trends (e.g. exports, GDP-related indicators) "
        "that reflect economic growth."
    )
    print(
        "- Check whether social indicators like life expectancy show steady improvement, "
        "which is typical of long-term development."
    )
    print(
        "- Use the correlation heatmap to identify clusters of indicators that move "
        "together (e.g. trade, GDP, energy use), which may inform feature engineering."
    )


# %% [markdown]
"""
### 3. Supervised Learning – Regression

We choose a **continuous target** based on economic activity.

Strategy:
1. Select a **regression target indicator** (prioritizing trade/GDP indicators).
2. Use all remaining indicators as **features**.
3. Train & evaluate:
   - **Linear Regression**
   - **KNN Regressor**
   - **Support Vector Regression (RBF kernel)**
   - **Random Forest Regressor**
4. Compare using **RMSE, MAE, R²**.
5. Interpret **feature importance** for the Random Forest model.
"""

# %%
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
    # Fallback: first available indicator
    return data.columns[0]


def prepare_supervised_datasets(
    data: pd.DataFrame, target_code: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Prepare train-test splits for regression on the chosen target.

    - Drop rows where the target is missing.
    - Use all other indicators as features.
    - StandardScaler will be applied inside the pipelines.
    """
    df_model = data.copy()
    df_model = df_model.dropna(subset=[target_code])

    y = df_model[target_code].values
    X = df_model.drop(columns=[target_code]).values
    feature_names = df_model.drop(columns=[target_code]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_names


def build_regression_models() -> dict:
    """
    Create a dict of regression models, each wrapped in a Pipeline that
    handles missing values and scaling.
    """
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    models = {
        "Linear Regression": Pipeline(base_steps + [("model", LinearRegression())]),
        "KNN Regressor": Pipeline(
            base_steps + [("model", KNeighborsRegressor(n_neighbors=5))]
        ),
        "SVR (RBF kernel)": Pipeline(
            base_steps
            + [
                (
                    "model",
                    SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
                )
            ]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                # no need to scale for trees, but kept simple
                ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
            ]
        ),
    }
    return models


def evaluate_regression_models(
    models: dict, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Fit and evaluate all regression models.

    Returns a DataFrame with RMSE, MAE, and R² on the test set.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}
        )

    results_df = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    return results_df


def plot_regression_feature_importance(
    rf_pipeline: Pipeline, feature_names: pd.Index, top_n: int = 10
) -> None:
    """
    Plot feature importance for the Random Forest regressor.
    """
    # The RandomForestRegressor is the last step in the pipeline
    rf_model: RandomForestRegressor = rf_pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="importance", y="feature", data=importance_df, palette="viridis"
    )
    plt.title("Top feature importances (Random Forest Regressor)")
    plt.tight_layout()
    plt.show()


# %% [markdown]
"""
### 4. Supervised Learning – Classification

We now create a **categorical target** describing *economic growth regimes*.

Approach:
1. Start from the same continuous target used for regression.
2. Compute the **year-over-year percentage growth**.
3. Discretize growth into:
   - **Low / Negative growth**
   - **Medium / Stable growth**
   - **High growth**
4. Train & evaluate classification models:
   - **Logistic Regression**
   - **KNN Classifier**
   - **SVM (RBF kernel)**
   - **Random Forest Classifier**
5. Evaluate using **Accuracy, Precision, Recall, F1-score** and **confusion matrices**.
"""

# %%
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

    # Drop first year (NaN growth)
    df_tmp = df_tmp.dropna(subset=["growth_rate"])

    # Initial fixed bins
    df_tmp["growth_category"] = pd.cut(
        df_tmp["growth_rate"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["Low/Negative", "Stable", "High"],
    )

    # If not all 3 classes are represented, use quantile-based bins
    if df_tmp["growth_category"].nunique() < 3:
        df_tmp["growth_category"] = pd.qcut(
            df_tmp["growth_rate"],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

    # Align with full feature matrix (same years, all indicators)
    df_features = data.loc[df_tmp.index].copy()
    df_cls = df_features.copy()
    df_cls["growth_rate"] = df_tmp["growth_rate"]
    df_cls["growth_category"] = df_tmp["growth_category"]

    # Drop rows with missing category (if any remain)
    df_cls = df_cls.dropna(subset=["growth_category"])

    return df_cls


def prepare_classification_data(
    df_cls: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, np.ndarray]:
    """
    Prepare X/y and train-test splits for classification.
    """
    y = df_cls["growth_category"].astype("category")
    X = df_cls.drop(columns=["growth_category", "growth_rate"])

    feature_names = X.columns
    X = X.values
    y_codes = y.cat.codes.values  # numeric labels
    class_names = y.cat.categories.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_codes, test_size=0.2, random_state=42, stratify=y_codes
    )

    return X_train, X_test, y_train, y_test, feature_names, class_names


def build_classification_models() -> dict:
    """
    Build classification models with pipelines for imputation + scaling.
    """
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    models = {
        "Logistic Regression": Pipeline(
            base_steps
            + [
                (
                    "model",
                    LogisticRegression(
                        multi_class="auto", max_iter=200, solver="lbfgs"
                    ),
                )
            ]
        ),
        "KNN Classifier": Pipeline(
            base_steps + [("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "SVM (RBF kernel)": Pipeline(
            base_steps
            + [
                (
                    "model",
                    SVC(kernel="rbf", C=10.0, gamma="scale"),
                )
            ]
        ),
        "Random Forest Classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        ),
    }
    return models


def evaluate_classification_models(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
) -> pd.DataFrame:
    """
    Fit and evaluate all classification models.

    Returns a DataFrame with accuracy, precision, recall and F1-score.
    Also prints detailed classification reports.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        results.append(
            {
                "model": name,
                "Accuracy": acc,
                "Precision(weighted)": prec,
                "Recall(weighted)": rec,
                "F1(weighted)": f1,
            }
        )

        print(f"\nClassification report for {name}:")
        print(
            classification_report(
                y_test, y_pred, target_names=class_names, zero_division=0
            )
        )

    results_df = pd.DataFrame(results).set_index("model").sort_values(
        "F1(weighted)", ascending=False
    )
    return results_df


def plot_confusion_matrices(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """
    Plot confusion matrices for all classification models.
    """
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    for i, (name, model) in enumerate(models.items(), start=1):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.subplot(n_rows, n_cols, i)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("True")

    plt.tight_layout()
    plt.show()


# %% [markdown]
"""
### 5. Unsupervised Learning – Clustering

We apply **K-Means clustering** on the socio-economic indicators to
discover groups of years with similar development profiles.

Steps:
1. Standardize all numeric indicators.
2. Apply **KMeans** with a small number of clusters (e.g., 3).
3. Summarize clusters by looking at **cluster centers** (mean indicator values).
4. Visualize in 2D using **PCA** to interpret cluster separation.
"""

# %%
def perform_kmeans_clustering(
    data: pd.DataFrame, n_clusters: int = 3
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run KMeans clustering on standardized indicators.

    Returns
    -------
    cluster_labels : pd.Series
        Cluster index for each year.
    cluster_centers_df : pd.DataFrame
        Cluster centers in original feature space (approximate, inverse-scaled).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Inverse-transform cluster centers to original scale for interpretation
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(
        centers_original, columns=data.columns, index=[f"Cluster {i}" for i in range(n_clusters)]
    )

    cluster_labels = pd.Series(clusters, index=data.index, name="cluster")

    # 2D PCA visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters,
        cmap="viridis",
        s=80,
        edgecolor="k",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clustering of Ethiopia socio-economic years (PCA projection)")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.show()

    print("\nCluster centers (approximate mean indicator values per cluster):")
    display(cluster_centers_df.iloc[:, :10])  # show first 10 indicators for brevity

    return cluster_labels, cluster_centers_df


# %% [markdown]
"""
### 6. Model Comparison & Analysis

We compare models in terms of:
- **Predictive performance** (metrics tables)
- **Model complexity** and **bias–variance tradeoff**:
  - Linear models: high bias, lower variance.
  - KNN: low bias, higher variance, sensitive to noise and `k`.
  - SVM with RBF: flexible decision boundaries, can overfit if not regularized.
  - Random Forest: strong performance, reduced variance via ensembling, but
    potential to overfit with many deep trees.

We also discuss **overfitting vs. underfitting**:
- Check performance gap between train and test sets.
- Look at whether very flexible models substantially outperform simpler ones.
"""

# %%
def summarize_bias_variance_and_overfitting(
    reg_results: pd.DataFrame, cls_results: pd.DataFrame
) -> None:
    """
    Print high-level discussion prompts based on model results.

    Note: For simplicity, we only used test metrics. In a full academic
    setting, you would also inspect **training metrics** and/or cross-validation
    curves to more directly observe overfitting/underfitting.
    """
    print("\n=== Regression model comparison ===")
    display(reg_results)

    print("\n=== Classification model comparison ===")
    display(cls_results)

    print(
        """
Interpretation guidelines:
- If a **simple model** (e.g. Linear Regression or Logistic Regression) performs
  similarly to more complex models, the relationship may be close to linear and
  **high-bias models** are sufficient.
- If **Random Forest** or **SVR/SVM** clearly outperform linear baselines, it
  suggests meaningful **non-linear structure** in the socio-economic indicators.
- Large performance differences across models, especially if complex models
  are very strong on small datasets, may indicate **overfitting**.
- Conversely, uniformly poor performance for all models might indicate
  **underfitting** or lack of predictive signal in the available indicators.
"""
    )


# %% [markdown]
"""
### 7. Results, Interpretation, and Conclusion

When you run this script, you should:
- Examine the **EDA outputs** to understand Ethiopia’s long-term development.
- Inspect **regression metrics** to see how well socio-economic indicators
  explain the chosen target (e.g., exports or GDP-related indicators).
- Inspect **classification metrics and confusion matrices** to understand how
  well different models distinguish between **low/medium/high growth regimes**.
- Look at **Random Forest feature importances** to identify indicators most
  strongly associated with growth.
- Study the **clustering results** to see whether distinct development
  phases emerge from the data.

Finally, reflect on:
- **Limitations**: single-country data, small sample size (years), missing
  indicators, ignoring time-series dependence, etc.
- **Possible improvements**: multi-country modeling, richer temporal models
  (e.g., RNNs, autoregressive models), more careful feature engineering,
  and incorporating domain knowledge from development economics.
- **Real-world applications**: forecasting export revenue, monitoring progress
  towards SDGs, identifying structural shifts in the economy, and supporting
  data-driven policy planning.
"""


# %% [markdown]
"""
### 8. Main execution

This `main()` function glues together:
- Data loading & preprocessing
- EDA
- Regression
- Classification
- Clustering
- Model comparison & interpretation
"""

# %%
def main():
    # 1. Data loading and preprocessing
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please ensure 'API_ETH_DS2_en_csv_v2_6515.csv' is in the working directory."
        )

    features_df, indicator_meta = load_and_prepare_dataset(DATA_PATH)

    print("=== 1. DATA UNDERSTANDING & PREPROCESSING ===")
    print("Years available:", features_df.index.min(), "to", features_df.index.max())
    print("Number of indicators used:", features_df.shape[1])

    # 2. EDA
    print("\n=== 2. EXPLORATORY DATA ANALYSIS (EDA) ===")
    perform_eda(features_df, indicator_meta)

    # 3. Supervised Learning – Regression
    print("\n=== 3A. SUPERVISED LEARNING – REGRESSION ===")
    target_code = choose_regression_target(features_df)
    target_name = indicator_meta.get(target_code, target_code)
    print(f"Chosen regression target: {target_code} – {target_name}")

    X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names_reg = (
        prepare_supervised_datasets(features_df, target_code)
    )
    reg_models = build_regression_models()
    reg_results = evaluate_regression_models(
        reg_models, X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )
    print("\nRegression performance (lower RMSE/MAE, higher R² are better):")
    display(reg_results)

    # Feature importance for Random Forest Regressor
    rf_reg_pipeline = reg_models["Random Forest Regressor"]
    plot_regression_feature_importance(rf_reg_pipeline, feature_names_reg)

    # 4. Supervised Learning – Classification
    print("\n=== 3B. SUPERVISED LEARNING – CLASSIFICATION ===")
    df_cls = create_growth_categories(features_df, target_code)
    (
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
        feature_names_cls,
        class_names,
    ) = prepare_classification_data(df_cls)

    cls_models = build_classification_models()
    cls_results = evaluate_classification_models(
        cls_models, X_train_cls, X_test_cls, y_train_cls, y_test_cls, class_names
    )
    print(
        "\nClassification performance (higher is better for all metrics):"
    )
    display(cls_results)
    plot_confusion_matrices(cls_models, X_test_cls, y_test_cls, class_names)

    # 5. Unsupervised Learning – Clustering
    print("\n=== 4. UNSUPERVISED LEARNING – KMEANS CLUSTERING ===")
    cluster_labels, cluster_centers_df = perform_kmeans_clustering(features_df)

    # 6. Model comparison & analysis
    print("\n=== 5. MODEL COMPARISON & ANALYSIS ===")
    summarize_bias_variance_and_overfitting(reg_results, cls_results)

    print("\n=== 6–7. RESULTS, INTERPRETATION, AND CONCLUSIONS ===")
    print(
        "See the printed comments, plots, and tables for detailed interpretation. "
        "Use them as a basis for a written project report (introduction, methods, "
        "results, discussion, conclusion, and future work)."
    )


if __name__ == "__main__":
    main()

