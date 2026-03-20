import numpy as np


def get_iqr_bounds(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


def cap_outliers(df, col, lower=None, upper=None):
    """Cap using pre-computed bounds when provided (transform), else recompute (fit)."""
    if lower is None or upper is None:
        lower, upper = get_iqr_bounds(df, col)
    df[col] = df[col].clip(lower, upper)
    return df


def log_transform(df, col):
    if (df[col] <= -1).any():
        return df
    df[col] = np.log1p(df[col])
    return df


def handle_outliers(df, skewness, model_type="linear"):
    """
    Fit-time handler. Records per-column decisions AND the exact IQR bounds
    used, so transform() can replay without touching test distribution.

    Returns (transformed_df, outlier_log) where outlier_log has:
      "capped"      -> {col: (lower, upper)}
      "transformed" -> [col, ...]
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    log = {"capped": {}, "transformed": []}

    for col in num_cols:
        if model_type == "tree":
            continue

        lower, upper = get_iqr_bounds(df, col)
        outlier_ratio = ((df[col] < lower) | (df[col] > upper)).mean()

        if col in skewness and skewness[col] > 1 and outlier_ratio > 0.15:
            if (df[col] > -1).all():
                df = log_transform(df, col)
                log["transformed"].append(col)
                continue

        # Store exact bounds so transform() doesn't recompute on test data
        df = cap_outliers(df, col, lower, upper)
        log["capped"][col] = (lower, upper)

    return df, log


def apply_outlier_log(df, outlier_log):
    """
    Transform-time handler. Uses the bounds stored during fit — no IQR
    recomputation on test data, which would be distribution leakage.
    """
    for col in outlier_log.get("transformed", []):
        if col in df.columns:
            df = log_transform(df, col)

    for col, (lower, upper) in outlier_log.get("capped", {}).items():
        if col in df.columns:
            df = cap_outliers(df, col, lower, upper)

    return df
