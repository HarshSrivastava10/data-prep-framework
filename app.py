import streamlit as st
import pandas as pd
from pipeline.data_cleaner import DataCleaner, split_data, detect_task

st.set_page_config(page_title="ML Cleaning Pipeline", layout="wide")
st.title("ML Data Cleaning Pipeline")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ------------------------------------------------------------------ #
    # Sidebar — all configuration lives here                              #
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.header("Pipeline config")

        target_col = st.selectbox("Target column", df.columns)
        model_type = st.selectbox("Model type", ["linear", "tree", "knn", "svm"])

        st.subheader("Splitting")
        split_enabled = st.checkbox("Enable train-test split", value=True)
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)

        st.subheader("Missing values")
        use_knn = st.checkbox("Use KNN imputation", value=False)
        missing_threshold = st.slider(
            "Drop column if missing > (%)", 10, 80, 30
        ) / 100

        st.subheader("Feature selection")
        top_features = st.slider("Max features to keep", 3, 20, 10)

        st.subheader("Diagnostics")
        step_timing = st.checkbox("Show step timings", value=False)

        run = st.button("Run cleaning pipeline", use_container_width=True)

    # ------------------------------------------------------------------ #
    # Tabs                                                                #
    # ------------------------------------------------------------------ #
    tab_profile, tab_clean, tab_report = st.tabs(
        ["Data profile", "Cleaned data", "Cleaning report"]
    )

    # --- Profiling tab (always shown, no pipeline needed) ---
    with tab_profile:
        st.subheader("Raw data preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Rows:** {df.shape[0]}  **Cols:** {df.shape[1]}")
            st.markdown("**Missing values**")
            missing = df.isnull().mean().mul(100).round(1)
            missing = missing[missing > 0].sort_values(ascending=False)
            if missing.empty:
                st.success("No missing values.")
            else:
                st.bar_chart(missing)

        with col2:
            st.markdown("**Column dtypes**")
            st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())

        st.markdown("**Numeric summary**")
        st.dataframe(df.describe().T)

        st.markdown("**Skewness**")
        skew = df.select_dtypes(include="number").skew().sort_values()
        st.bar_chart(skew)

    # ------------------------------------------------------------------ #
    # Pipeline execution                                                  #
    # ------------------------------------------------------------------ #
    if run:
        config = {
            "model_type": model_type,
            "use_knn": use_knn,
            "missing_threshold": missing_threshold,
            "feature_selection": True,
            "top_features": top_features,
            "step_timing": step_timing,
            "test_split": {
                "enabled": split_enabled,
                "test_size": test_size,
                "random_state": 42,
            },
        }

        cleaner = DataCleaner(config)
        train_clean = test_clean = cleaned_df = None

        try:
            if split_enabled:
                config["task"] = detect_task(df[target_col])
                X_train, X_test, y_train, y_test = split_data(df, target_col, config)
                train_df = pd.concat([X_train, y_train], axis=1)
                test_df  = pd.concat([X_test,  y_test],  axis=1)
                train_clean = cleaner.fit_transform(train_df, target_col)
                test_clean  = cleaner.transform(test_df)
            else:
                cleaned_df = cleaner.fit_transform(df, target_col)

            # sync detected task back to plain config dict
            config["task"] = cleaner.config.task

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

        # --- Cleaned data tab ---
        with tab_clean:
            if split_enabled:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Train (cleaned)")
                    st.dataframe(train_clean.head())
                with c2:
                    st.subheader("Test (cleaned)")
                    st.dataframe(test_clean.head())
            else:
                st.subheader("Cleaned data")
                st.dataframe(cleaned_df.head())

            # Before / after column comparison
            st.subheader("Before vs after — numeric stats")
            raw_num   = df.select_dtypes(include="number").describe().T
            clean_src = train_clean if split_enabled else cleaned_df
            clean_num = clean_src.select_dtypes(include="number").describe().T
            common    = raw_num.index.intersection(clean_num.index)
            if not common.empty:
                cmp = pd.concat(
                    [raw_num.loc[common, ["mean","std"]].add_suffix("_raw"),
                     clean_num.loc[common, ["mean","std"]].add_suffix("_clean")],
                    axis=1,
                ).round(3)
                st.dataframe(cmp)

            # Feature importance
            st.subheader("Feature importance")
            importance = cleaner.get_feature_importance()
            if importance is not None:
                st.bar_chart(importance.sort_values(ascending=False).head(top_features))
            else:
                st.info("Feature importance not available.")

            st.info(f"Detected task: **{config['task']}**")

            # Download cleaned data
            st.subheader("Download cleaned data")
            if split_enabled:
                st.download_button(
                    "Download train CSV",
                    train_clean.to_csv(index=False),
                    "train_clean.csv",
                )
                st.download_button(
                    "Download test CSV",
                    test_clean.to_csv(index=False),
                    "test_clean.csv",
                )
            else:
                st.download_button(
                    "Download cleaned CSV",
                    cleaned_df.to_csv(index=False),
                    "cleaned_data.csv",
                )

            # Save fitted pipeline
            st.subheader("Save fitted pipeline")
            if st.button("Save pipeline to disk (pipeline.joblib)"):
                try:
                    cleaner.save("pipeline.joblib")
                    st.success("Pipeline saved as pipeline.joblib")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        # --- Report tab ---
        with tab_report:
            report = cleaner.get_report()
            st.text(report.summary())

            st.download_button(
                "Download HTML report",
                report.to_html(),
                "cleaning_report.html",
                mime="text/html",
            )
