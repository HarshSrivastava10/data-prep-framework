import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def drop_id_like_columns(df):
    """Drop near-unique integer/object columns that behave like row IDs."""
    id_cols = [
        col for col in df.columns
        if df[col].nunique() / len(df) >= 0.95
        and df[col].dtype.kind in ("i", "u", "O")
    ]
    return df.drop(columns=id_cols), id_cols


def variance_threshold(df, threshold=0.01):
    selector = VarianceThreshold(threshold)
    reduced = selector.fit_transform(df)
    selected_cols = df.columns[selector.get_support()]
    return pd.DataFrame(reduced, columns=selected_cols, index=df.index)


def correlation_filter(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop_cols), drop_cols


def model_feature_importance(X, y, task):
    """
    Compute feature importance. Uses SHAP (TreeExplainer) when available
    because it corrects for the high-cardinality bias in RF's built-in
    feature_importances_. Falls back to RF importances if shap is not installed.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if task == "regression" or y.dtype.kind in "fc":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For classifiers shap_values is a list (one array per class) — take mean abs
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    except ImportError:
        importance = pd.Series(model.feature_importances_, index=X.columns)

    return importance.sort_values(ascending=False)


def select_top_features(X, importance, top_n=10):
    selected = importance.head(top_n).index
    return X[selected]


def feature_selection(df, target_col, task="classification", top_n=10):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Include bool cols (pd.get_dummies output in pandas >= 1.5)
    X = X.select_dtypes(include=["number", "bool"])
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Drop ID-like columns before importance ranking
    X, dropped_ids = drop_id_like_columns(X)
    if dropped_ids:
        import warnings
        warnings.warn(f"Dropped ID-like columns: {dropped_ids}")

    X = X.dropna(axis=1, how="all").fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No numeric/bool features available for feature selection.")

    X = variance_threshold(X)
    X, _ = correlation_filter(X)

    if X.shape[1] == 0:
        raise ValueError("No features left after variance and correlation filtering.")

    importance = model_feature_importance(X, y, task)

    top_n = min(top_n, X.shape[1])
    X = select_top_features(X, importance, top_n)
    X = X.loc[y.index]

    return pd.concat([X, y], axis=1), importance