"""
End-to-end linear regression learning script.
Covers:
1) Simple linear regression from scratch (normal equation for 1 feature)
2) Simple and multiple linear regression with scikit-learn
3) Advanced variants: Polynomial regression, Ridge, Lasso, ElasticNet
4) Validation: train/test split and k-fold cross-validation
5) Assumption checks: linearity, independence, homoscedasticity, normality, multicollinearity
Dataset source (web):
https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
"""

from __future__ import annotations  # Allows using newer type hint syntax on older Python versions

import warnings  # Used to suppress noisy but harmless sklearn/statsmodels UserWarnings
from dataclasses import dataclass  # Provides a clean way to define data-holding classes
from typing import Dict, Tuple  # Type hints for function signatures

# --- Numerical & Data libraries ---
import numpy as np   # Core numerical operations (arrays, math, random)
import pandas as pd  # Tabular data loading and manipulation

# --- Statistical modelling (for assumption tests) ---
import statsmodels.api as sm  # OLS regression used only for Breusch-Pagan test
from scipy import stats       # Shapiro-Wilk normality test lives here

# --- scikit-learn: preprocessing ---
from sklearn.compose import ColumnTransformer    # Applies different transforms to different columns
from sklearn.impute import SimpleImputer         # Fills missing values (median for numbers, mode for text)
from sklearn.preprocessing import (
    OneHotEncoder,      # Converts categorical strings → binary (0/1) columns
    PolynomialFeatures, # Generates x², x*y, etc. for polynomial regression
    StandardScaler,     # Scales numeric features to mean=0, std=1 (important for Ridge/Lasso)
)

# --- scikit-learn: models ---
from sklearn.linear_model import (
    ElasticNet,       # Combines Ridge (L2) + Lasso (L1) regularization
    Lasso,            # L1 regularization: can shrink some coefficients to exactly 0 (feature selection)
    LinearRegression, # Ordinary least squares with no regularization
    Ridge,            # L2 regularization: penalizes large coefficients, prevents overfitting
)

# --- scikit-learn: evaluation & pipeline ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,            # Splits data into k equal folds for cross-validation
    cross_val_score,  # Runs CV automatically and returns scores for each fold
    train_test_split, # Splits the dataset into training and testing subsets
)
from sklearn.pipeline import Pipeline  # Chains preprocessor + model into one reusable object

# --- statsmodels: assumption diagnostics ---
from statsmodels.stats.diagnostic import het_breuschpagan        # Tests for heteroscedasticity
from statsmodels.stats.stattools import durbin_watson             # Tests for autocorrelation in residuals
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Measures multicollinearity

# Suppress harmless convergence and category warnings so output stays clean
warnings.filterwarnings("ignore", category=UserWarning)

# --- Dataset URL ---
# Using a publicly available insurance CSV so anyone can run this script without local files
DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)


# ---------------------------------------------------------------------------
# DATA CLASS: stores all evaluation metrics for a single model in one place
# ---------------------------------------------------------------------------
@dataclass
class RegressionMetrics:
    model_name: str       # Name/label of the model (e.g. "Ridge(alpha=1.0)")
    mae: float            # Mean Absolute Error: average of |actual - predicted|
    mse: float            # Mean Squared Error: average of (actual - predicted)²; penalises large errors more
    rmse: float           # Root MSE: same unit as target, easier to interpret than MSE
    r2: float             # R²: proportion of variance explained (1.0 = perfect, 0 = baseline mean)
    adjusted_r2: float    # Adjusted R²: penalises adding useless features; better for model comparison
    cv_rmse_mean: float   # Average RMSE across all k CV folds (more reliable than single test-set score)
    cv_rmse_std: float    # Std dev of CV RMSE: how stable the model is across different data splits


# ---------------------------------------------------------------------------
# HELPER: Adjusted R² formula
# ---------------------------------------------------------------------------
def adjusted_r2_score(r2: float, n_samples: int, n_features: int) -> float:
    """
    Compute adjusted R² to fairly compare models with different numbers of features.

    Plain R² always increases (or stays the same) when you add more features, even
    useless ones. Adjusted R² corrects for this by penalising model complexity.

    Formula:
        adj_R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where n = number of samples, p = number of features.
    """
    if n_samples <= n_features + 1:
        # Can't compute: would cause division by zero or negative denominator
        return float("nan")

    # (n - 1) / (n - p - 1): the correction factor that grows as p increases
    return 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))


# ---------------------------------------------------------------------------
# STEP 1: Load the dataset from the web
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Download a real-world dataset from the web."""
    print("\n[1/7] Loading dataset from web source...")

    # pd.read_csv works directly on URLs — no manual download needed
    df = pd.read_csv(DATA_URL)

    print(f"Dataset shape: {df.shape}")   # (rows, columns)
    print("Columns:", list(df.columns))   # Should be: age, sex, bmi, children, smoker, region, charges
    return df


# ---------------------------------------------------------------------------
# STEP 2: Implement simple linear regression manually (educational)
# ---------------------------------------------------------------------------
def simple_linear_regression_from_scratch(df: pd.DataFrame) -> None:
    """
    Teach linear regression fundamentals using closed-form (normal equation) formulas.

    We use a single feature (age) to predict charges.
    The closed-form solution avoids gradient descent and gives the exact optimal slope/intercept.
    """
    print("\n[2/7] Simple Linear Regression from scratch (age -> charges)")

    # Extract the single feature and target as plain NumPy arrays
    x = df["age"].to_numpy(dtype=float)      # Input feature: age
    y = df["charges"].to_numpy(dtype=float)  # Target variable: medical charges

    # --- Normal equation for slope and intercept ---
    x_mean = x.mean()  # Mean of feature values
    y_mean = y.mean()  # Mean of target values

    # Numerator: sum of element-wise products of (x - x̄)(y - ȳ)  → covariance(x, y) * n
    numerator = np.sum((x - x_mean) * (y - y_mean))

    # Denominator: sum of squared deviations from the mean  → variance(x) * n
    denominator = np.sum((x - x_mean) ** 2)

    # Slope (β₁): how much y changes for a 1-unit increase in x
    slope = numerator / denominator

    # Intercept (β₀): predicted y when x = 0 (derived from the means)
    intercept = y_mean - slope * x_mean

    # Generate predictions using the fitted line: ŷ = β₀ + β₁ * x
    y_pred = intercept + slope * x

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"y = {intercept:.2f} + {slope:.2f} * age")  # Print the learned equation
    print(f"MSE: {mse:.2f}")
    print(f"R^2: {r2:.4f}")  # Low R² expected; age alone doesn't explain all variance


# ---------------------------------------------------------------------------
# HELPER: Build the preprocessing pipeline for mixed feature types
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    """
    Create separate preprocessing pipelines for numeric and categorical columns,
    then combine them with ColumnTransformer so sklearn can handle both at once.

    Why separate pipelines?
    - Numeric features need imputation (fill NaN with median) + scaling (StandardScaler).
    - Categorical features need imputation (fill NaN with mode) + one-hot encoding.
    """

    # Automatically detect which columns are numeric vs. text/object
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # --- Numeric pipeline ---
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Replace NaN with column median (robust to outliers)
            ("scaler", StandardScaler()),                   # Standardise: (x - mean) / std → mean=0, std=1
        ]
    )

    # --- Categorical pipeline ---
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Replace NaN with the most common category
            (
                "onehot",
                OneHotEncoder(
                    drop="first",           # Drop first category column to avoid dummy variable trap (multicollinearity)
                    handle_unknown="ignore" # Silently ignore categories seen at predict time but not at train time
                ),
            ),
        ]
    )

    # Combine both pipelines: apply numeric_pipeline to numeric columns, categorical_pipeline to the rest
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


# ---------------------------------------------------------------------------
# HELPER: Fit a model pipeline and compute all evaluation metrics
# ---------------------------------------------------------------------------
def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[RegressionMetrics, np.ndarray]:
    """
    Train the pipeline on X_train/y_train, evaluate on X_test/y_test,
    and also run 5-fold cross-validation on the training set for a more
    robust performance estimate.

    Returns both the metrics object and the test-set predictions (for residual plots etc.).
    """

    # Fit the full pipeline (preprocessor + model) on training data only
    pipeline.fit(X_train, y_train)

    # Predict on the held-out test set — pipeline applies the same transforms automatically
    y_pred = pipeline.predict(X_test)

    # --- Compute standard regression metrics ---
    mae = mean_absolute_error(y_test, y_pred)   # Average absolute error (same unit as target)
    mse = mean_squared_error(y_test, y_pred)    # Average squared error (heavily penalises big mistakes)
    rmse = np.sqrt(mse)                         # Square-root of MSE → interpretable in original units
    r2 = r2_score(y_test, y_pred)               # Proportion of variance explained (0–1, higher is better)

    # Count features AFTER preprocessing (polynomial features expand count dramatically)
    transformed_feature_count = pipeline.named_steps["preprocessor"].fit_transform(X_train).shape[1]

    # Adjusted R² accounts for the number of features so we can fairly compare models
    adj_r2 = adjusted_r2_score(r2, len(y_test), transformed_feature_count)

    # --- 5-Fold Cross-Validation on the training set ---
    # KFold splits training data into 5 equal parts; each part is used once as validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",  # sklearn uses negative scores so higher = better internally
        n_jobs=None,
    )

    # Bundle all metrics into the dataclass; negate CV scores to get positive RMSE values
    metrics = RegressionMetrics(
        model_name=model_name,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        adjusted_r2=adj_r2,
        cv_rmse_mean=float(-cv_scores.mean()),  # Negate to get positive RMSE
        cv_rmse_std=float(cv_scores.std()),     # Spread across folds: low std = stable model
    )

    return metrics, y_pred


# ---------------------------------------------------------------------------
# STEP 3: Build, train, and compare all regression model variants
# ---------------------------------------------------------------------------
def run_models(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Pipeline]:
    """
    Split data, build five regression variants, evaluate each, and return a
    sorted comparison table alongside the baseline pipeline for assumption checks.
    """
    print("\n[3/7] Building train/test splits and regression models...")

    # Separate features (X) from the target variable (y)
    X = df.drop(columns=["charges"])  # All columns except the target
    y = df["charges"]                  # Target: medical insurance charges

    # 80% of data goes to training, 20% held out for final evaluation
    # random_state=42 ensures the same split every run (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build the shared preprocessor (same object reused across all models)
    preprocessor, _, _ = build_preprocessor(X)

    # --- Define all model variants to compare ---
    model_builders = {
        # Ordinary least squares: minimises sum of squared residuals with no penalty
        "MultipleLinearRegression": LinearRegression(),

        # Polynomial regression: adds squared & interaction terms (x², x*y, …) before fitting OLS
        # degree=2 means we include original features AND all their pairwise products/squares
        "PolynomialRegression(degree=2)": Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linreg", LinearRegression()),
            ]
        ),

        # Ridge (L2): adds α * Σβ² penalty; shrinks large coefficients but keeps all features
        # Good when many features are weakly correlated with the target
        "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),

        # Lasso (L1): adds α * Σ|β| penalty; can zero out irrelevant features entirely
        # Effectively does automatic feature selection
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, random_state=42, max_iter=5000),

        # ElasticNet: weighted mix of L1 and L2; l1_ratio=0.5 means equal blend
        # Useful when features are correlated and you want some sparsity
        "ElasticNet(alpha=0.01,l1_ratio=0.5)": ElasticNet(
            alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000
        ),
    }

    all_metrics: list[RegressionMetrics] = []
    predictions: Dict[str, np.ndarray] = {}
    baseline_pipeline: Pipeline | None = None  # We'll save the plain OLS pipeline for assumption checks

    for name, model in model_builders.items():
        # Wrap each model together with the preprocessor into a single Pipeline object
        # This ensures preprocessing params are learned on train data only (no data leakage)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        metrics, y_pred = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        all_metrics.append(metrics)
        predictions[name] = y_pred  # Save predictions for potential residual analysis

        # Keep a reference to the plain linear regression pipeline for assumption diagnostics
        if name == "MultipleLinearRegression":
            baseline_pipeline = pipeline

    # Convert list of metric dataclasses to a DataFrame and sort by RMSE (ascending = better)
    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics]).sort_values(by="rmse")

    print("\nModel comparison (lower RMSE is better):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    if baseline_pipeline is None:
        raise RuntimeError("Baseline model pipeline was not created.")

    return metrics_df, predictions, X_train, y_train, X_test, y_test, baseline_pipeline


# ---------------------------------------------------------------------------
# STEP 4: Statistical assumption checks for linear regression
# ---------------------------------------------------------------------------
def assumption_checks(
    baseline_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Validate the five key assumptions of linear regression on the baseline OLS model.
    Violating these assumptions can make predictions or p-values unreliable.
    """
    print("\n[4/7] Checking linear regression assumptions on baseline model...")

    # Get predictions on the test set to compute residuals
    y_pred_test = baseline_pipeline.predict(X_test)

    # Residuals = actual - predicted  (ideally random noise with no pattern)
    residuals = y_test.to_numpy() - y_pred_test

    # ------------------------------------------------------------------
    # ASSUMPTION 1: LINEARITY
    # If the model is truly linear, residuals should be random (no pattern) relative to fitted values.
    # Correlation near 0 is a proxy check: a strong correlation suggests a non-linear relationship.
    # ------------------------------------------------------------------
    corr = np.corrcoef(y_pred_test, residuals)[0, 1]  # Pearson correlation between predictions and residuals
    print(f"Linearity proxy (corr(pred, residual)): {corr:.4f}")
    # Ideal: close to 0. A large positive/negative value signals missed non-linearity.

    # ------------------------------------------------------------------
    # ASSUMPTION 2: INDEPENDENCE OF RESIDUALS
    # Durbin-Watson tests whether consecutive residuals are correlated (autocorrelation).
    # Range: 0–4. ~2 means no autocorrelation; <1.5 or >2.5 warrants investigation.
    # ------------------------------------------------------------------
    dw = durbin_watson(residuals)
    print(f"Durbin-Watson statistic (ideal ~2): {dw:.4f}")

    # ------------------------------------------------------------------
    # ASSUMPTION 3: NORMALITY OF RESIDUALS
    # Linear regression inference (p-values, confidence intervals) assumes normally distributed residuals.
    # Shapiro-Wilk is one of the most powerful tests for this.
    # p > 0.05 → fail to reject normality (residuals are approximately normal).
    # ------------------------------------------------------------------
    sample_for_shapiro = residuals
    if len(residuals) > 5000:
        # Shapiro-Wilk is computationally expensive for large samples; use a random subset
        rng = np.random.default_rng(42)
        sample_for_shapiro = rng.choice(residuals, size=5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_for_shapiro)
    print(f"Shapiro-Wilk p-value for residual normality: {shapiro_p:.6f}")

    # ------------------------------------------------------------------
    # ASSUMPTION 4: HOMOSCEDASTICITY (constant variance of residuals)
    # If residuals fan out or shrink as fitted values grow, variance is not constant (heteroscedasticity).
    # Breusch-Pagan regresses squared residuals on the predictors.
    # p > 0.05 → fail to reject constant variance.
    # We use statsmodels OLS here because sklearn doesn't expose residuals the same way.
    # ------------------------------------------------------------------
    # Re-transform training features using the already-fitted preprocessor
    transformed_train = baseline_pipeline.named_steps["preprocessor"].fit_transform(X_train)
    transformed_train_df = pd.DataFrame(transformed_train)  # Convert sparse/numpy array → DataFrame

    # Add intercept column (constant) required by statsmodels OLS
    transformed_train_with_const = sm.add_constant(transformed_train_df)

    # Fit OLS using statsmodels (only needed to get its residuals for Breusch-Pagan)
    ols_model = sm.OLS(y_train.to_numpy(), transformed_train_with_const).fit()

    # Breusch-Pagan test: returns (LM statistic, LM p-value, F statistic, F p-value)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(ols_model.resid, ols_model.model.exog)
    print(f"Breusch-Pagan p-value (homoscedasticity): {bp_pvalue:.6f}")

    # ------------------------------------------------------------------
    # ASSUMPTION 5: NO MULTICOLLINEARITY
    # Variance Inflation Factor (VIF) measures how much a feature's variance is inflated
    # because it's linearly predictable from other features.
    # VIF = 1: no correlation. VIF > 5: moderate concern. VIF > 10: serious concern.
    # ------------------------------------------------------------------
    vif_frame = pd.DataFrame({"feature": transformed_train_with_const.columns})
    vif_values = []
    exog = transformed_train_with_const.to_numpy(dtype=float)  # VIF function expects a numpy array

    for i in range(exog.shape[1]):
        # variance_inflation_factor(exog_matrix, column_index) computes VIF for that column
        vif_values.append(variance_inflation_factor(exog, i))

    vif_frame["vif"] = vif_values

    # Remove the constant row (VIF for intercept is not meaningful)
    vif_frame = vif_frame[vif_frame["feature"] != "const"].sort_values(by="vif", ascending=False)

    print("\nTop 10 highest VIF values:")
    print(vif_frame.head(10).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # --- Quick interpretation guide ---
    print("\nAssumption quick interpretation:")
    print("- Linearity: residual-prediction correlation should be near 0.")
    print("- Independence: Durbin-Watson near 2 is generally good.")
    print("- Normality: Shapiro p > 0.05 suggests residual normality.")
    print("- Homoscedasticity: Breusch-Pagan p > 0.05 suggests constant variance.")
    print("- Multicollinearity: VIF > 5 (or >10) is often considered problematic.")


# ---------------------------------------------------------------------------
# STEP 5: Show how to make a prediction on a real row
# ---------------------------------------------------------------------------
def manual_prediction_demo(df: pd.DataFrame, baseline_pipeline: Pipeline) -> None:
    """Demonstrate how to feed a raw data row through the trained pipeline and get a prediction."""
    print("\n[5/7] Manual prediction demo using trained baseline model...")

    # Take the first row of the dataset as an example input
    # iloc[[0]] (double brackets) keeps the result as a DataFrame (shape 1×n) instead of a Series
    # The pipeline expects a 2D input, so this is important
    example = df.drop(columns=["charges"]).iloc[[0]].copy()

    # pipeline.predict() automatically applies the same preprocessing, then predicts
    pred = baseline_pipeline.predict(example)[0]  # [0] extracts the scalar from the 1-element array

    print("Example input row:")
    print(example.to_string(index=False))
    print(f"Predicted charges: {pred:,.2f}")


# ---------------------------------------------------------------------------
# STEP 6: Print a concise learning summary
# ---------------------------------------------------------------------------
def learning_summary(metrics_df: pd.DataFrame) -> None:
    """Print the best-performing model and a recap of what this script demonstrated."""
    print("\n[6/7] Learning summary")

    # After sorting by RMSE in run_models(), iloc[0] is the row with the smallest (best) RMSE
    best_model = metrics_df.iloc[0]
    print(f"Best model by RMSE: {best_model['model_name']}")

    print("What you learned in this script:")
    print("1. Simple linear regression math from scratch.")
    print("2. Data preprocessing for mixed feature types.")
    print("3. Multiple regression variants and regularization.")
    print("4. Holdout + cross-validation based model validation.")
    print("5. Statistical assumption checks for trustable inference.")


# ---------------------------------------------------------------------------
# MAIN: Orchestrates all steps in order
# ---------------------------------------------------------------------------
def main() -> None:
    print("Linear Regression End-to-End Educational Walkthrough")

    # Step 1 — Download and inspect the dataset
    df = load_dataset()

    # Step 2 — Demonstrate the maths behind simple linear regression
    simple_linear_regression_from_scratch(df)

    # Step 3 — Build all models; the _ discards the predictions dict (not needed here)
    metrics_df, _, X_train, y_train, X_test, y_test, baseline_pipeline = run_models(df)

    # Step 4 — Validate statistical assumptions on the plain OLS model
    assumption_checks(baseline_pipeline, X_train, y_train, X_test, y_test)

    # Step 5 — Show a concrete prediction on a single real row
    manual_prediction_demo(df, baseline_pipeline)

    # Step 6 — Summarise what was covered
    learning_summary(metrics_df)

    print("\n[7/7] Completed.")


# Standard Python entry-point guard: only run main() when the script is executed directly,
# not when it's imported as a module by another script.
if __name__ == "__main__":
    main()
