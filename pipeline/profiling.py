import pandas as pd

def load_data(file_path):
  if file_path.endswith(".csv"):
      return pd.read_csv(file_path)
  elif file_path.endswith(".xlsx"):
      return pd.read_excel(file_path)

def basic_info(df):
  return{
      "shape": df.shape,
      "columns": df.columns.tolist(),
      "dtypes": df.dtypes.astype(str).to_dict()
  }

# Missing Values Analysis
def missing_values(df):
  missing = df.isnull().sum()
  percent = (missing/len(df)) * 100

  return pd.DataFrame({
      "missing_count":missing,
      "missing_percent":percent
  }).sort_values(by="missing_percent", ascending=False)

# Numerical Statistics
def numerical_summary(df):
  num_df = df.select_dtypes(include=['number'])
  return num_df.describe().T

#Categorical Analysis
def categorical_summary(df):
  cat_cols = df.select_dtypes(include=['object', 'category']).columns

  summary = {}

  for col in cat_cols:
    summary[col] = {
        "unique_values": df[col].nunique(),
        "top_values": df[col].value_counts().head(5).to_dict()
    }
  return summary

# Outlier Dectection
def outlier_detection(df):
  outlier_report = {}

  num_cols = df.select_dtypes(include=['number']).columns

  for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_report[col] = outlier_count

  return outlier_report


# Skewness Detection
def skewness_detection(df):
  num_cols = df.select_dtypes(include=['number']).columns
  if len(num_cols) == 0:
    return {}
  return df[num_cols].skew().to_dict()

# Combined Profiling Function
def profile_data(df):
  missing_df = missing_values(df)

  high_missing_cols = missing_df[missing_df["missing_percent"] > 30].index.tolist()

  constant_cols = [col for col in df.columns if df[col].nunique() <= 1]

  type_counts = df.dtypes.value_counts().astype(str).to_dict()

  return{
      "basic_info": basic_info(df),
      "missing_values": missing_df,
      "numerical_summary": numerical_summary(df),
      "categorical_summary": categorical_summary(df),
      "outlier_detection": outlier_detection(df),
      "skewness_detection": skewness_detection(df),

      "high_missing_cols": high_missing_cols,
      "constant_cols": constant_cols,
      "type_counts": type_counts
  }