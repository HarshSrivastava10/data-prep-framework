import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def get_categorical_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns


def label_encode(df, cols):
    le_dict = {}
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict


def one_hot_encode(df, cols):
    return pd.get_dummies(df, columns=cols, drop_first=True)


def frequency_encoding(df, cols):
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
    return df


def fit_encoding(df, model_type, ordinal_cols=None, y=None):
    """
    ordinal_cols : dict  {col_name: [cat_low, ..., cat_high]}
    y            : Series  required for target encoding (high-cardinality, non-tree)
    """
    if ordinal_cols is None:
        ordinal_cols = {}

    encoders = {
        "label":     {},
        "frequency": {},
        "target":    {},
        "ordinal":   {},
        "onehotcols": [],
    }

    # Step 1 — ordinal encoding (applied before everything else)
    for col, order in ordinal_cols.items():
        if col in df.columns:
            oe = OrdinalEncoder(categories=[order],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1)
            df[col] = oe.fit_transform(df[[col]])
            encoders["ordinal"][col] = oe

    cat_cols = [c for c in get_categorical_cols(df) if c not in ordinal_cols]

    if model_type == "tree":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders["label"][col] = le
        return df, encoders

    low_card  = [c for c in cat_cols if df[c].nunique() < 10]
    high_card = [c for c in cat_cols if df[c].nunique() >= 10]

    # One-hot for low cardinality
    if low_card:
        df = one_hot_encode(df, low_card)
        encoders["onehotcols"] = df.columns.tolist()

    # Target encoding for high cardinality when y is available, else frequency
    for col in high_card:
        if y is not None:
            try:
                from sklearn.preprocessing import TargetEncoder
                te = TargetEncoder(smooth="auto")
                df[col] = te.fit_transform(df[[col]], y).ravel()
                encoders["target"][col] = te
                continue
            except ImportError:
                pass  # sklearn < 1.3 — fall through to frequency encoding

        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq).fillna(0)
        encoders["frequency"][col] = freq

    return df, encoders


def transform_encoding(df, encoders, model_type):
    # Ordinal — replay fitted OrdinalEncoders
    for col, oe in encoders.get("ordinal", {}).items():
        if col in df.columns:
            df[col] = oe.transform(df[[col]])

    cat_cols = [c for c in get_categorical_cols(df)
                if c not in encoders.get("ordinal", {})]

    if model_type == "tree":
        for col, le in encoders["label"].items():
            if col in df.columns:
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                df[col] = df[col].astype(str).map(mapping).fillna(0)
        return df

    # Target encoding replay
    for col, te in encoders.get("target", {}).items():
        if col in df.columns:
            df[col] = te.transform(df[[col]]).ravel()

    # Frequency encoding replay
    for col, freq in encoders["frequency"].items():
        if col in df.columns:
            df[col] = df[col].map(freq).fillna(0)

    # One-hot: re-encode remaining low-cardinality cats then align columns
    ohe_cols = [c for c in cat_cols
                if c not in encoders["frequency"]
                and c not in encoders.get("target", {})]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    for col in encoders["onehotcols"]:
        if col not in df.columns:
            df[col] = 0
    df = df[encoders["onehotcols"]]

    return df


# Legacy helper kept for backward compatibility
def encode_features(df, model_type="linear"):
    cat_cols = get_categorical_cols(df)
    if len(cat_cols) == 0:
        return df
    if model_type == "tree":
        df, _ = label_encode(df, cat_cols)
        return df
    low_card  = [c for c in cat_cols if df[c].nunique() < 10]
    high_card = [c for c in cat_cols if df[c].nunique() >= 10]
    if low_card:
        df = one_hot_encode(df, low_card)
    if high_card:
        df = frequency_encoding(df, high_card)
    return df