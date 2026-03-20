import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pipeline.encoding import fit_encoding, transform_encoding
from pipeline.missing import handle_missing_values
from pipeline.outliers import handle_outliers, apply_outlier_log
from pipeline.profiling import profile_data
from pipeline.feature_selection import feature_selection
from pipeline.config_schema import CleaningConfig
from pipeline.exception import FitBeforeTransformError, SerializationError
from pipeline.validation import validate_input
from pipeline.report import CleaningReport


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Modular tabular data cleaning pipeline.

    Inherits from BaseEstimator + TransformerMixin so it works inside
    sklearn.pipeline.Pipeline and is compatible with GridSearchCV.
    """

    def __init__(self, config: dict):
        self.config = CleaningConfig(**config)
        self.imputers = {}
        self.scaler = None
        self.encoders = {}
        self.selected_features = None
        self.outlier_log = {"capped": {}, "transformed": []}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df, target_col):
        self.report = CleaningReport()
        validate_input(df, target_col)
        self.target_col = target_col

        X = df.drop(columns=[target_col])
        y = df[target_col]

        detected_task = detect_task(y)
        self.config.task = detected_task
        self.report.original_shape = df.shape
        self.report.task_detected = detected_task

        def _t(label, fn):
            """Run fn(), optionally record elapsed time."""
            if self.config.step_timing:
                t0 = time.perf_counter()
                result = fn()
                self.report.step_timings[label] = round(time.perf_counter() - t0, 4)
                return result
            return fn()

        # --- profiling ---
        self.profile = _t("profiling", lambda: profile_data(X))
        skewness = self.profile["skewness_detection"]

        # --- missing values ---
        def _missing():
            return handle_missing_values(
                X, skewness,
                use_knn=self.config.use_knn,
                threshold=self.config.missing_threshold,
            )
        result = _t("missing", _missing)
        X, missing_log, missing_info = result
        self.imputers = missing_info
        self.report.dropped_columns = missing_info["dropped_cols"]
        self.report.imputed_columns = {
            **{c: "median/mean" for c in missing_info["num_imputers"]},
            **{c: "mode/constant" for c in missing_info["cat_imputers"]},
            **({} if missing_info["knn_imputer"] is None
               else {"[knn]": "KNNImputer"}),
        }

        # --- outliers ---
        def _outliers():
            return handle_outliers(X, skewness, self.config.model_type)
        X, self.outlier_log = _t("outliers", _outliers)
        self.report.outlier_actions = (
            {c: "log_transform" for c in self.outlier_log["transformed"]}
            | {c: "capped" for c in self.outlier_log["capped"]}
        )

        # --- encoding ---
        def _encoding():
            return fit_encoding(
                X,
                self.config.model_type,
                ordinal_cols=dict(self.config.ordinal_cols),
                y=y,
            )
        X, self.encoders = _t("encoding", _encoding)
        self.report.encoded_columns = {
            **{c: "label"     for c in self.encoders["label"]},
            **{c: "frequency" for c in self.encoders["frequency"]},
            **{c: "target"    for c in self.encoders.get("target", {})},
            **{c: "ordinal"   for c in self.encoders.get("ordinal", {})},
            **{c: "one_hot"   for c in self.encoders["onehotcols"]},
        }

        # --- feature selection ---
        if self.config.feature_selection:
            df_temp = pd.concat([X, y], axis=1)

            def _feat_sel():
                return feature_selection(
                    df_temp,
                    self.target_col,
                    self.config.task,
                    top_n=self.config.top_features,
                )
            df_selected, importance = _t("feature_selection", _feat_sel)
            self.selected_features = df_selected.drop(columns=[self.target_col]).columns
            self.feature_importance = importance
            self.report.selected_features = list(self.selected_features)
            X = df_selected.drop(columns=[self.target_col])

        # --- scaling ---
        self.scaling_cols = X.select_dtypes(include=["number"]).columns

        def _build_scaler():
            if self.config.model_type == "tree":
                return None
            if self.config.model_type in ("knn",):
                return MinMaxScaler()
            return StandardScaler()

        self.scaler = _build_scaler()
        if self.scaler is not None and len(self.scaling_cols) > 0:
            self.scaler.fit(X[self.scaling_cols])
            X[self.scaling_cols] = self.scaler.transform(X[self.scaling_cols])

        self.report.final_shape = X.shape
        self.X_processed = X
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, df):
        if not hasattr(self, "profile"):
            raise FitBeforeTransformError("Must call fit() before transform().")

        X = df.drop(columns=[self.target_col])

        # missing
        X = X.drop(columns=self.imputers["dropped_cols"], errors="ignore")
        for col, imputer in self.imputers["num_imputers"].items():
            if col in X.columns:
                X[col] = imputer.transform(X[[col]])
        for col, value in self.imputers["cat_imputers"].items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        if self.imputers.get("knn_imputer") is not None:
            num_cols = X.select_dtypes(include=["number"]).columns
            X[num_cols] = self.imputers["knn_imputer"].transform(X[num_cols])

        # outliers — replay fit decisions with stored bounds
        X = apply_outlier_log(X, self.outlier_log)

        # encoding
        X = transform_encoding(X, self.encoders, self.config.model_type)

        # feature selection
        if self.selected_features is not None:
            for col in self.selected_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.selected_features]

        # scaling
        valid_cols = [c for c in self.scaling_cols if c in X.columns]
        if self.scaler is not None and valid_cols:
            X[valid_cols] = self.scaler.transform(X[valid_cols])

        return pd.concat([X, df[self.target_col]], axis=1)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def fit_transform(self, df, target_col, **_):
        self.fit(df, target_col)
        return pd.concat([self.X_processed, df[target_col]], axis=1)

    def get_report(self):
        if hasattr(self, "report"):
            return self.report
        raise FitBeforeTransformError("Must call fit() before get_report().")

    def get_feature_importance(self):
        return getattr(self, "feature_importance", None)

    # ------------------------------------------------------------------
    # serialization  (Step 9)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the fitted pipeline to disk using joblib."""
        try:
            import joblib
            joblib.dump(self, path)
        except Exception as e:
            raise SerializationError(f"Failed to save pipeline to '{path}': {e}") from e

    @classmethod
    def load(cls, path: str) -> "DataCleaner":
        """Load a previously saved pipeline from disk."""
        try:
            import joblib
            obj = joblib.load(path)
            if not isinstance(obj, cls):
                raise SerializationError(f"Loaded object is not a DataCleaner.")
            return obj
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"Failed to load pipeline from '{path}': {e}") from e

    # ------------------------------------------------------------------
    # sklearn / repr
    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        return {"config": self.config.model_dump()}

    def set_params(self, **params):
        if "config" in params:
            self.config = CleaningConfig(**params["config"])
        return self

    def __repr__(self):
        fitted = hasattr(self, "profile")
        return (
            f"DataCleaner("
            f"model={self.config.model_type}, "
            f"task={self.config.task}, "
            f"knn={self.config.use_knn}, "
            f"top_features={self.config.top_features}, "
            f"fitted={fitted})"
        )


# ------------------------------------------------------------------
# module-level helpers
# ------------------------------------------------------------------
def split_data(df, target_col, config):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if not config["test_split"]["enabled"]:
        return X, None, y, None
    stratify = y if config["task"] == "classification" else None
    return train_test_split(
        X, y,
        test_size=config["test_split"]["test_size"],
        random_state=config["test_split"]["random_state"],
        stratify=stratify,
    )


def detect_task(y):
    if y.dtype.kind in "fc":
        return "regression"
    if y.nunique() <= 20:
        return "classification"
    return "regression"
