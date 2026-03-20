from pydantic import BaseModel, Field
from typing import Literal, Dict, List


class TestSplitConfig(BaseModel):
    enabled: bool = True
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = 42


class CleaningConfig(BaseModel):
    model_type: Literal["linear", "tree", "knn", "svm"]
    task: Literal["classification", "regression", "auto"] = "auto"

    # --- missing values ---
    feature_selection: bool = True
    use_knn: bool = False                        # Step 3: wire KNN imputation
    missing_threshold: float = Field(0.3, description="Drop column if missing > this fraction")

    # --- outliers ---
    outlier_strategy: Literal["cap", "log", "skip"] = "cap"

    # --- feature selection ---
    top_features: int = Field(10, ge=1, description="Max features kept after selection")  # Step 5

    # --- encoding ---
    # Step 6: ordinal_cols maps col name -> ordered category list (low → high)
    ordinal_cols: Dict[str, List[str]] = {}

    # --- diagnostics ---
    step_timing: bool = False                    # Step 8: benchmark each fit step

    test_split: TestSplitConfig = TestSplitConfig()
