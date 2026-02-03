## Ethiopia Socio-Economic ML Project

This project provides an end-to-end machine learning pipeline using World Bank
World Development Indicators (WDI) data for **Ethiopia**
(`API_ETH_DS2_en_csv_v2_6515.csv`).

It is designed as a **university-level ML course project** to illustrate:

- **Data understanding & preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Supervised learning** (regression & classification)
- **Unsupervised learning** (clustering)
- **Model comparison, bias–variance tradeoff, and interpretation**

### 1. Files

- `API_ETH_DS2_en_csv_v2_6515.csv` – Ethiopia WDI dataset (World Bank export)
- `ethiopia_socioeconomic_ml_project.py` – main Python script (notebook-style)
- `ethiopia_socioeconomic_ml_project.ipynb` – Jupyter notebook version
- `eth_ml/` – reusable package (`data`, `models`, `utils`)
- `train_models.py` – offline training and model saving script
- `app.py` – Streamlit web application
- `requirements.txt` – Python dependencies

### 2. Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the CSV file `API_ETH_DS2_en_csv_v2_6515.csv` is in the **same directory**
as the project root (same folder as `app.py`).

### 3. Running the offline ML project

Run the end-to-end pipeline from the command line:

```bash
python ethiopia_socioeconomic_ml_project.py
```

The script will:

1. **Load & preprocess** the WDI Ethiopia dataset (handle metadata rows, reshape
   to a Year × Indicator matrix, select numeric features).
2. Perform **EDA**:
   - Summary statistics for indicators
   - Correlation heatmap for high-variance indicators
   - Trend plots for key socio-economic variables
3. Run **regression**:
   - Choose a continuous economic target
   - Train Linear Regression, KNN Regressor, SVR (RBF), Random Forest Regressor
   - Compare models with RMSE, MAE, and R²
   - Show Random Forest feature importances
4. Run **classification**:
   - Derive low/medium/high **growth regimes** from year-over-year changes
   - Train Logistic Regression, KNN Classifier, SVM (RBF), Random Forest Classifier
   - Evaluate with Accuracy, Precision, Recall, F1-score
   - Plot confusion matrices
5. Run **unsupervised learning**:
   - KMeans clustering on standardized indicators
   - Visualize clusters in 2D using PCA
   - Inspect cluster centers (development profiles)
6. Provide **guidance on model comparison** (bias–variance, overfitting vs.
   underfitting) and **interpretation** of results in the context of Ethiopia’s
   development trajectory.

### 4. Training and saving models for deployment

Before running the Streamlit app, train and persist the models:

```bash
python train_models.py
```

This script will:

- Load and preprocess the Ethiopia dataset using `eth_ml.data`
- Train:
  - RandomForestRegressor (regression task)
  - RandomForestClassifier (classification task)
  - StandardScaler + KMeans + PCA (clustering)
- Save:
  - `models/regression_model.pkl`
  - `models/classification_model.pkl`
  - `models/clustering_model.pkl`
  - `models/regression_metrics.json`
  - `models/classification_metrics.json`
  - `models/training_metadata.json`

All paths are defined relative to the project root, so the code is
environment-independent as long as you run commands from the project folder.

### 5. Running the Streamlit web application

#### 5.1 Local deployment

After training the models (`python train_models.py`), start the app locally:

```bash
streamlit run app.py
```

The app will:

- Load the persisted regression, classification, and clustering models
- Provide a **sidebar** where you can:
  - Select a historical year from the Ethiopia dataset
  - Upload a CSV row of socio-economic indicators
  - Manually adjust key indicators (top features)
- Display:
  - Regression predictions for the chosen target (e.g., exports or GDP-related)
  - Growth regime classification (Low/Medium/High)
  - Model performance summaries (RMSE, MAE, R², Accuracy, F1)
  - Feature importance plots
  - Actual vs. predicted trends over time
  - Cluster visualization in PCA space, highlighting the current input

The app includes basic error handling for missing models and invalid uploads
and uses environment-independent paths (`eth_ml.utils.ROOT_DIR` and
`eth_ml.utils.get_models_dir()`).

#### 5.2 Streamlit Cloud deployment

1. Push this project to a public Git repository (e.g., GitHub).
2. On Streamlit Cloud, create a new app and point it to:
   - **Repository**: your Git repo URL
   - **Branch**: e.g., `main`
   - **Main file**: `app.py`
3. In the repo, ensure:
   - `requirements.txt` is present (includes `streamlit`, `joblib`, etc.).
   - `API_ETH_DS2_en_csv_v2_6515.csv` is committed at the project root.
4. Option A (recommended): run `python train_models.py` locally and commit the
   resulting `models/` directory to the repo, so the app can read models
   directly.
5. Option B: configure a one-time initialization script on Streamlit Cloud
   (e.g., via the “Advanced settings” → “Pre-run script”) that runs:

   ```bash
   python train_models.py
   ```

   so models are generated when the app first starts.

### 6. Using it as a course project

Students can:

- Run the script as-is to see the full pipeline.
- Open `ethiopia_socioeconomic_ml_project.py` in a notebook-aware editor
  (e.g., VS Code) and treat `# %%` cells as Jupyter cells.
- Use `ethiopia_socioeconomic_ml_project.ipynb` for a fully notebook-based
  workflow.
- Extend the project by:
  - Trying alternative targets (e.g., life expectancy, energy use).
  - Adjusting model hyperparameters and evaluating effects on bias–variance.
  - Adding more advanced time-series models.
  - Comparing Ethiopia with other countries (by loading additional WDI exports).


