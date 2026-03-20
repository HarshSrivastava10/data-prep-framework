def split_columns(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return num_cols, cat_cols


def handle_numeric(df, num_cols, skewness, threshold=30):
    from sklearn.impute import SimpleImputer
    imputers = {}
    dropped_cols = []

    for col in num_cols:
        missing_pct = df[col].isnull().mean() * 100

        if missing_pct == 0:
            continue

        if missing_pct == 100 or missing_pct > threshold:
            df = df.drop(columns=[col])
            dropped_cols.append(col)
            continue

        if col in skewness and abs(skewness[col]) > 1:
            imputer = SimpleImputer(strategy="median")
        else:
            imputer = SimpleImputer(strategy="mean")

        df[col] = imputer.fit_transform(df[[col]])
        imputers[col] = imputer

    return df, imputers, dropped_cols


def handle_categorical(df, cat_cols, threshold=30):
    imputers = {}

    for col in cat_cols:
        missing_pct = df[col].isnull().mean() * 100

        if missing_pct == 100:
            continue

        fill_value = "Missing" if missing_pct > threshold else (
            df[col].mode()[0] if not df[col].mode().empty else "Missing"
        )

        df[col] = df[col].fillna(fill_value)
        imputers[col] = fill_value

    return df, imputers


def knn_imputation(df, num_cols):
    """Fit a KNNImputer and return (transformed_df, fitted_imputer)
    so the imputer can be stored and replayed in transform()."""
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df, imputer


def handle_missing_values(df, skewness, use_knn=False, threshold=30):
    num_cols, cat_cols = split_columns(df)

    df, num_imputers, dropped_cols = handle_numeric(df, num_cols, skewness, threshold)

    _, cat_cols = split_columns(df)
    df, cat_imputers = handle_categorical(df, cat_cols, threshold)

    knn_imputer = None
    if use_knn:
        num_cols_remaining = df.select_dtypes(include=["number"]).columns
        if len(num_cols_remaining) > 0:
            df, knn_imputer = knn_imputation(df, num_cols_remaining)

    log = {
        "num_imputed": list(num_imputers.keys()),
        "cat_imputed": list(cat_imputers.keys()),
        "dropped_cols": dropped_cols,
    }
    return df, log, {
        "num_imputers": num_imputers,
        "cat_imputers": cat_imputers,
        "dropped_cols": dropped_cols,
        "knn_imputer": knn_imputer,   # None when use_knn=False
    }
