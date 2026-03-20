import pandas as pd
from pipeline.exception import InsufficientDataError

def validate_input(df, target_col) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    if len(df) < 10:
        raise InsufficientDataError("Dataset must have at least 10 rows for processing.")
    
    if df[target_col].nunique() < 2:
        raise InsufficientDataError("Target column must have at least 2 unique values.")
    
    all_null_cols = [c for c in df.columns if df[c].isnull().all()]

    if all_null_cols:
        import warnings
        warnings.warn(f"The following columns are entirely null and will be dropped: {all_null_cols}")

        