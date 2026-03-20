"""
feature_engineering.py
-----------------------
Domain-specific feature construction that runs BEFORE DataCleaner.

Two sections:
  1. Generic helpers — ratio/interaction/binning transforms that work on any
     dataset. Wire them by passing column names.
  2. Dataset-specific examples — Titanic shown as a template. Copy and adapt
     for your own dataset.

Usage in app.py (or a notebook):
    from pipeline.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(dataset="titanic")   # or dataset="generic"
    df = fe.transform(df)
    cleaner = DataCleaner(config)
    cleaner.fit_transform(df, target_col)
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Stateless transformer — no fit step needed because all operations are
    deterministic functions of the input columns.
    """

    def __init__(self, dataset: str = "generic"):
        """
        dataset : str
            "generic"  — only generic helpers are applied (safe for any CSV)
            "titanic"  — Titanic-specific features added on top
            Add your own dataset name and a matching _engineer_<name> method.
        """
        self.dataset = dataset.lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Always run generic helpers first
        df = self._generic(df)

        # Dataset-specific layer
        method = f"_engineer_{self.dataset}"
        if hasattr(self, method):
            df = getattr(self, method)(df)

        return df

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _generic(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @staticmethod
    def add_ratio(df, numerator_col, denominator_col, new_col=None):
        """Safe ratio — returns 0 where denominator is 0."""
        name = new_col or f"{numerator_col}_per_{denominator_col}"
        denom = df[denominator_col].replace(0, np.nan)
        df[name] = (df[numerator_col] / denom).fillna(0)
        return df

    @staticmethod
    def add_interaction(df, col_a, col_b, new_col=None):
        """Multiplicative interaction term."""
        name = new_col or f"{col_a}_x_{col_b}"
        df[name] = df[col_a] * df[col_b]
        return df

    @staticmethod
    def add_log(df, col, new_col=None):
        """Log1p transform as a new column — useful for right-skewed features."""
        name = new_col or f"log_{col}"
        df[name] = np.log1p(df[col].clip(lower=0))
        return df

    @staticmethod
    def add_binned(df, col, bins, labels, new_col=None):
        """Bin a numeric column into ordinal categories."""
        name = new_col or f"{col}_bin"
        df[name] = pd.cut(df[col], bins=bins, labels=labels, right=False)
        return df

    # ------------------------------------------------------------------
    # Titanic — example dataset-specific engineering
    # ------------------------------------------------------------------
    def _engineer_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        # Family size — captures group dynamics better than SibSp/Parch alone
        if {"SibSp", "Parch"}.issubset(df.columns):
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

        # Title — encodes social status / gender more granularly than Sex alone
        if "Name" in df.columns:
            df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
            rare = ["Lady","Countess","Capt","Col","Don","Dr",
                    "Major","Rev","Sir","Jonkheer","Dona"]
            df["Title"] = df["Title"].replace(rare, "Rare")
            df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

        # Deck — extracted from Cabin; most missing values become "Unknown"
        if "Cabin" in df.columns:
            df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

        # Fare per person — corrects for group tickets where Fare is shared
        if {"Fare", "FamilySize"}.issubset(df.columns):
            df = self.add_ratio(df, "Fare", "FamilySize", "FarePerPerson")

        # Age binned — captures non-linear age effects (child, adult, elderly)
        if "Age" in df.columns:
            df = self.add_binned(
                df, "Age",
                bins=[0, 12, 18, 35, 60, 120],
                labels=["child", "teen", "young_adult", "adult", "senior"],
                new_col="AgeBand",
            )

        return df
