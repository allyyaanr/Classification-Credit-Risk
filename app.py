import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD DATA & MODEL
# =========================================================

DATA_PATH = "dataset1.csv"
MODEL_PATH = "credit_risk_model.pkl"
RESULT_PATH = "logistic_results.pkl"


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Dataset tidak dapat dibaca.\n\nError: {e}"
    )

    st.stop()


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    logistic_pipeline = load_model()

except Exception as e:

    st.error(
        f"Model tidak dapat dimuat.\n\nError: {e}"
    )

    st.stop()


# ============================================================
# 6. LOAD MODEL EVALUATION
# ============================================================

@st.cache_resource
def load_results():

    return joblib.load(RESULT_PATH)


try:

    logistic_results = load_results()

except Exception:

    logistic_results = None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Credit Risk")
st.sidebar.write("Logistic Regression Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "📈 Exploratory Data Analysis",
        "🤖 Model Performance",
        "🔮 Credit Prediction"
    ]
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "📊 Overview":

    st.title("💳 Credit Risk Classification")

    st.markdown(
        """
        ### About This Project

        This project uses **Logistic Regression** to classify
        customers based on their credit risk.

        The model predicts whether a customer belongs to:

        - 🟢 **Good Credit Risk**
        - 🔴 **Bad Credit Risk**

        The application provides dataset exploration,
        model performance evaluation, and credit risk prediction.
        """
    )

    st.divider()

    # =====================================================
    # DATASET SOURCE
    # =====================================================

    st.subheader("📂 Dataset Source")

    st.markdown(
        """
        This project uses the **South German Credit** dataset
        for credit risk classification.

        The dataset contains customer financial and personal
        information that is used to predict credit risk.
        """
    )

    st.link_button(
        "🔗 View Dataset on Kaggle",
        "https://www.kaggle.com/datasets/sid321axn/south-german-credit-updated"
    )

    st.divider()

    # -----------------------------------------------------
    # Dataset Information
    # -----------------------------------------------------

    st.subheader("📂 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    with col3:
        if "credit_risk" in df.columns:
            st.metric(
                "Target Variable",
                "credit_risk"
            )
        else:
            st.metric(
                "Target Variable",
                "Not Found"
            )

    st.divider()

    # -----------------------------------------------------
    # Target Distribution
    # -----------------------------------------------------

    if "credit_risk" in df.columns:

        st.subheader("✪ Credit Risk Distribution")

        risk_counts = df["credit_risk"].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.dataframe(
                risk_counts.rename("Count"),
                width=700
            )

        with col2:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            sns.countplot(
                data=df,
                x="credit_risk",
                ax=ax
            )

            ax.set_xlabel("Credit Risk")
            ax.set_ylabel("Number of Customers")

            st.pyplot(fig)

            plt.close(fig)

    st.divider()

    # -----------------------------------------------------
    # Dataset Preview
    # -----------------------------------------------------

    st.subheader("✪ Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# =========================================================

elif page == "📈 Exploratory Data Analysis":

    st.title("📈 Exploratory Data Analysis")

    st.markdown(
        """
        This section provides an overview of the dataset,
        including missing values, descriptive statistics,
        target distribution, feature distributions,
        and correlations.
        """
    )

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.subheader("✪ Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    with col4:
        st.metric(
            "Total Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.subheader("✪ Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # MISSING VALUES
    # =====================================================

    st.subheader("✪ Missing Values")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if missing_df.empty:

        st.success(
            "There are no missing values in the dataset."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=missing_df,
            x="Missing Values",
            y="Column",
            ax=ax
        )

        ax.set_title(
            "Missing Values by Column"
        )

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # =====================================================
    # DESCRIPTIVE STATISTICS
    # =====================================================

    st.subheader("✪ Descriptive Statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # CREDIT RISK DISTRIBUTION
    # =====================================================

    if "credit_risk" in df.columns:

        st.subheader("✪ Credit Risk Distribution")

        col1, col2 = st.columns(2)

        with col1:

            risk_counts = (
                df["credit_risk"]
                .value_counts()
            )

            st.dataframe(
                risk_counts.rename("Count"),
                use_container_width=True
            )

        with col2:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            sns.countplot(
                data=df,
                x="credit_risk",
                ax=ax
            )

            ax.set_title(
                "Credit Risk Distribution"
            )

            ax.set_xlabel(
                "Credit Risk"
            )

            ax.set_ylabel(
                "Number of Customers"
            )

            st.pyplot(fig)

            plt.close(fig)

    st.divider()

    # =====================================================
    # NUMERICAL FEATURES
    # =====================================================

    st.subheader("✪ Numerical Feature Distribution")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numerical_columns) > 0:

        selected_numeric = st.selectbox(
            "Select Numerical Feature",
            numerical_columns
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            data=df,
            x=selected_numeric,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_numeric}"
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No numerical features found."
        )

    st.divider()

    # =====================================================
    # CATEGORICAL FEATURES
    # =====================================================

    st.subheader("✪ Categorical Feature Distribution")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Jangan tampilkan target sebagai feature
    if "credit_risk" in categorical_columns:

        categorical_features = [
            col for col in categorical_columns
            if col != "credit_risk"
        ]

    else:

        categorical_features = categorical_columns

    if len(categorical_features) > 0:

        selected_categorical = st.selectbox(
            "Select Categorical Feature",
            categorical_features
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.countplot(
            data=df,
            x=selected_categorical,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_categorical}"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No categorical features found. "
        )

    st.divider()

    # =====================================================
    # CORRELATION MATRIX
    # =====================================================

    st.subheader("✪ Correlation Matrix")

    correlation_data = df.select_dtypes(
        include=np.number
    )

    if correlation_data.shape[1] > 1:

        correlation_matrix = (
            correlation_data.corr()
        )

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Feature Correlation Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "Not enough numerical features "
            "to create a correlation matrix."
        )


# =========================================================
# PAGE 3 — MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.markdown(
        """
        This section evaluates the performance of the
        **Logistic Regression** classification model.
        """
    )

    # =====================================================
    # CHECK RESULTS
    # =====================================================

    if logistic_results is None:

        st.error(
            "logistic_results.pkl could not be loaded."
        )

        st.stop()

    # =====================================================
    # DEBUG — OPTIONAL
    # =====================================================

    # Uncomment jika ingin melihat isi dictionary
    #
    # st.write(logistic_results.keys())

    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("✪ Evaluation Metrics")

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    available_metrics = {}

    for metric in metric_names:

        if metric in logistic_results:

            available_metrics[metric] = (
                logistic_results[metric]
            )

    if len(available_metrics) == 0:

        st.warning(
            "Metric is not found in "
            "logistic_results.pkl."
        )

        st.write(
            "Available keys:"
        )

        st.write(
            list(logistic_results.keys())
        )

    else:

        columns = st.columns(
            len(available_metrics)
        )

        for col, (metric, value) in zip(
            columns,
            available_metrics.items()
        ):

            with col:

                st.metric(
                    metric,
                    f"{value:.4f}"
                )

    st.divider()

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("✪ Confusion Matrix")

    if "confusion_matrix" in logistic_results:

        cm = np.array(
            logistic_results["confusion_matrix"]
        )

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Good", "Bad"],
            yticklabels=["Good", "Bad"],
            ax=ax
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)

        st.caption(
            "Rows = Actual, Columns = Predicted"
        )

    else:

        st.warning(
            "Confusion Matrix is not available "
            "in logistic_results.pkl."
        )

    st.divider()

    # =====================================================
    # ROC CURVE
    # =====================================================

    st.subheader("✪ ROC Curve")

    if "roc_curve" in logistic_results:

        roc_data = logistic_results[
            "roc_curve"
        ]

        fpr = np.array(
            roc_data["fpr"]
        )

        tpr = np.array(
            roc_data["tpr"]
        )

        if "roc_auc" in logistic_results:

            roc_auc = logistic_results[
                "roc_auc"
            ]

        else:

            roc_auc = auc(
                fpr,
                tpr
            )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "ROC Curve"
        )

        ax.legend(
            loc="lower right"
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "ROC curve data is not yet available "
            "in logistic_results.pkl."
        )

    st.divider()

    # =====================================================
    # RESULT KEYS
    # =====================================================

    with st.expander(
        "🔧 Debug: Available Result Keys"
    ):

        st.write(
            list(logistic_results.keys())
        )


# =========================================================
# PAGE 4 — CREDIT PREDICTION
# =========================================================

elif page == "🔮 Credit Prediction":

    st.title("🔮 Credit Risk Prediction")

    st.write(
        """
        Enter the applicant's information below to predict
        their credit risk using the trained Logistic Regression model.
        """
    )

    st.divider()

    # =====================================================
    # MODEL FEATURES
    # =====================================================

    model_features = logistic_pipeline.feature_names_in_

    st.caption(
        f"Model menggunakan {len(model_features)} input features."
    )

    # =====================================================
    # INPUT FORM
    # =====================================================

    with st.form("credit_prediction_form"):

        st.subheader("👤 Applicant Information")

        col1, col2 = st.columns(2)

        # =================================================
        # LEFT COLUMN
        # =================================================

        with col1:

            # ---------------------------------------------
            # STATUS
            # ---------------------------------------------

            status_options = (
                df["status"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            status = st.selectbox(
                "Checking Account Status",
                options=status_options
            )

            # ---------------------------------------------
            # DURATION
            # ---------------------------------------------

            duration_numeric = pd.to_numeric(
                df["duration"],
                errors="coerce"
            )

            duration = st.number_input(
                "Credit Duration (Months)",
                min_value=float(duration_numeric.min()),
                max_value=float(duration_numeric.max()),
                value=float(duration_numeric.median()),
                step=1.0
            )

            # ---------------------------------------------
            # CREDIT HISTORY
            # ---------------------------------------------

            credit_history_options = (
                df["credit_history"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            credit_history = st.selectbox(
                "Credit History",
                options=credit_history_options
            )

            # ---------------------------------------------
            # PURPOSE
            # ---------------------------------------------

            purpose_options = (
                df["purpose"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            purpose = st.selectbox(
                "Purpose",
                options=purpose_options
            )

            # ---------------------------------------------
            # AMOUNT
            # ---------------------------------------------

            amount_numeric = pd.to_numeric(
                df["amount"],
                errors="coerce"
            )

            amount = st.number_input(
                "Credit Amount",
                min_value=float(amount_numeric.min()),
                max_value=float(amount_numeric.max()),
                value=float(amount_numeric.median()),
                step=100.0
            )

            # ---------------------------------------------
            # SAVINGS
            # ---------------------------------------------

            savings_options = (
                df["savings"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            savings = st.selectbox(
                "Savings Account",
                options=savings_options
            )

            # ---------------------------------------------
            # EMPLOYMENT DURATION
            # ---------------------------------------------

            employment_options = (
                df["employment_duration"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            employment_duration = st.selectbox(
                "Employment Duration",
                options=employment_options
            )

            # ---------------------------------------------
            # INSTALLMENT RATE
            # ---------------------------------------------

            installment_options = (
                df["installment_rate"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            installment_rate = st.selectbox(
                "Installment Rate",
                options=installment_options
            )

            # ---------------------------------------------
            # PERSONAL STATUS / SEX
            # ---------------------------------------------

            personal_status_options = (
                df["personal_status_sex"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            personal_status_sex = st.selectbox(
                "Personal Status / Sex",
                options=personal_status_options
            )

            # ---------------------------------------------
            # OTHER DEBTORS
            # ---------------------------------------------

            other_debtors_options = (
                df["other_debtors"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            other_debtors = st.selectbox(
                "Other Debtors",
                options=other_debtors_options
            )

        # =================================================
        # RIGHT COLUMN
        # =================================================

        with col2:

            # ---------------------------------------------
            # PRESENT RESIDENCE
            # ---------------------------------------------

            residence_options = (
                df["present_residence"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            present_residence = st.selectbox(
                "Present Residence Duration",
                options=residence_options
            )

            # ---------------------------------------------
            # PROPERTY
            # ---------------------------------------------

            property_options = (
                df["property"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            property_value = st.selectbox(
                "Property",
                options=property_options
            )

            # ---------------------------------------------
            # AGE
            # ---------------------------------------------

            age_numeric = pd.to_numeric(
                df["age"],
                errors="coerce"
            )

            age = st.number_input(
                "Age",
                min_value=float(age_numeric.min()),
                max_value=float(age_numeric.max()),
                value=float(age_numeric.median()),
                step=1.0
            )

            # ---------------------------------------------
            # OTHER INSTALLMENT PLANS
            # ---------------------------------------------

            other_installment_options = (
                df["other_installment_plans"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            other_installment_plans = st.selectbox(
                "Other Installment Plans",
                options=other_installment_options
            )

            # ---------------------------------------------
            # HOUSING
            # ---------------------------------------------

            housing_options = (
                df["housing"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            housing = st.selectbox(
                "Housing",
                options=housing_options
            )

            # ---------------------------------------------
            # NUMBER OF EXISTING CREDITS
            # ---------------------------------------------

            credits_options = (
                df["number_credits"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            number_credits = st.selectbox(
                "Number of Existing Credits",
                options=credits_options
            )

            # ---------------------------------------------
            # JOB
            # ---------------------------------------------

            job_options = (
                df["job"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            job = st.selectbox(
                "Job",
                options=job_options
            )

            # ---------------------------------------------
            # PEOPLE LIABLE
            # ---------------------------------------------

            people_liable_options = (
                df["people_liable"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            people_liable = st.selectbox(
                "People Liable / Dependents",
                options=people_liable_options
            )

            # ---------------------------------------------
            # TELEPHONE
            # ---------------------------------------------

            telephone_options = (
                df["telephone"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            telephone = st.selectbox(
                "Telephone",
                options=telephone_options
            )

            # ---------------------------------------------
            # FOREIGN WORKER
            # ---------------------------------------------

            foreign_worker_options = (
                df["foreign_worker"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            foreign_worker = st.selectbox(
                "Foreign Worker",
                options=foreign_worker_options
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔮 Predict Credit Risk",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame({
            "status": [status],
            "duration": [duration],
            "credit_history": [credit_history],
            "purpose": [purpose],
            "amount": [amount],
            "savings": [savings],
            "employment_duration": [employment_duration],
            "installment_rate": [installment_rate],
            "personal_status_sex": [personal_status_sex],
            "other_debtors": [other_debtors],
            "present_residence": [present_residence],
            "property": [property_value],
            "age": [age],
            "other_installment_plans": [other_installment_plans],
            "housing": [housing],
            "number_credits": [number_credits],
            "job": [job],
            "people_liable": [people_liable],
            "telephone": [telephone],
            "foreign_worker": [foreign_worker]
        })

        # -------------------------------------------------
        # CHECK FEATURES
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in model_features
            if feature not in input_data.columns
        ]

        if len(missing_features) > 0:

            st.error(
                "❌ Feature berikut tidak ditemukan "
                f"dalam input: {missing_features}"
            )

            st.stop()

        # -------------------------------------------------
        # ARRANGE COLUMNS
        # -------------------------------------------------

        input_data = input_data[
            list(model_features)
        ]

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            prediction = logistic_pipeline.predict(
                input_data
            )[0]

            probabilities = (
                logistic_pipeline
                .predict_proba(input_data)[0]
            )

            classes = logistic_pipeline.classes_

            good_index = list(classes).index(
                "good"
            )

            bad_index = list(classes).index(
                "bad"
            )

            probability_good = (
                probabilities[good_index]
            )

            probability_bad = (
                probabilities[bad_index]
            )

        except Exception as e:

            st.error(
                "❌ Terjadi error saat melakukan prediction."
            )

            st.code(str(e))

            st.stop()

        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == "bad":

            st.error(
                "🔴 HIGH CREDIT RISK — BAD",
                icon="⚠️"
            )

            st.write(
                """
                Based on the information provided, the applicant
                is predicted to have a **Bad Credit Risk**.
                """
            )

        else:

            st.success(
                "🟢 LOW CREDIT RISK — GOOD",
                icon="✅"
            )

            st.write(
                """
                Based on the information provided, the applicant
                is predicted to have a **Good Credit Risk**.
                """
            )

        # =================================================
        # PROBABILITY
        # =================================================

        st.subheader("🎯 Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🟢 Good Credit Probability",
                f"{probability_good:.2%}"
            )

        with col2:

            st.metric(
                "🔴 Bad Credit Probability",
                f"{probability_bad:.2%}"
            )

        # =================================================
        # PROBABILITY CHART
        # =================================================

        probability_df = pd.DataFrame({
            "Credit Risk": [
                "Good",
                "Bad"
            ],
            "Probability": [
                probability_good,
                probability_bad
            ]
        })

        st.bar_chart(
            probability_df.set_index(
                "Credit Risk"
            )
        )

        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.subheader("🔎 Applicant Information")

        display_input = input_data.copy()

        st.dataframe(
            display_input.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )



# # =========================================================
# # PAGE 4 — CREDIT PREDICTION
# # =========================================================

# elif page == "🔮 Credit Prediction":

#     st.title("🔮 Credit Risk Prediction")

#     st.markdown(
#         """
#         Enter customer information below to predict
#         the customer's credit risk.
#         """
#     )

#     st.warning(
#         """
#         ⚠️ Make sure the names and order of the features you enter
#         match the features used when training the model.
#         """
#     )

#     # =====================================================
#     # CHECK MODEL FEATURES
#     # =====================================================

#     try:

#         if hasattr(
#             logistic_pipeline,
#             "feature_names_in_"
#         ):

#             model_features = (
#                 logistic_pipeline.feature_names_in_
#             )

#         else:

#             model_features = None

#     except Exception:

#         model_features = None

#     # =====================================================
#     # SHOW MODEL FEATURES
#     # =====================================================

#     if model_features is not None:

#         with st.expander(
#             "🔍 View Model Features"
#         ):

#             st.write(
#                 list(model_features)
#             )

#     # =====================================================
#     # INPUT FORM
#     # =====================================================

#     st.subheader("Customer Information")

#     # -----------------------------------------------------
#     # Basic numeric inputs
#     # -----------------------------------------------------

#     col1, col2 = st.columns(2)

#     with col1:

#         age = st.number_input(
#             "Age",
#             min_value=18,
#             max_value=100,
#             value=30
#         )

#         duration = st.number_input(
#             "Duration",
#             min_value=1,
#             max_value=100,
#             value=12
#         )

#         credit_amount = st.number_input(
#             "Credit Amount",
#             min_value=0,
#             value=1000
#         )

#         installment_rate = st.number_input(
#             "Installment Rate",
#             min_value=1,
#             max_value=10,
#             value=4
#         )

#     with col2:

#         residence_duration = st.number_input(
#             "Residence Duration",
#             min_value=1,
#             max_value=10,
#             value=2
#         )

#         existing_credits = st.number_input(
#             "Existing Credits",
#             min_value=1,
#             max_value=10,
#             value=1
#         )

#         people_liable = st.number_input(
#             "People Liable",
#             min_value=1,
#             max_value=10,
#             value=1
#         )

#     st.divider()

#     # -----------------------------------------------------
#     # Categorical inputs
#     # -----------------------------------------------------

#     st.subheader("Customer Category")

#     col1, col2 = st.columns(2)

#     with col1:

#         checking_account = st.selectbox(
#             "Checking Account",
#             [
#                 "little",
#                 "moderate",
#                 "rich",
#                 "no checking"
#             ]
#         )

#         saving_account = st.selectbox(
#             "Saving Account",
#             [
#                 "little",
#                 "moderate",
#                 "quite rich",
#                 "rich",
#                 "no known savings"
#             ]
#         )

#         employment_duration = st.selectbox(
#             "Employment Duration",
#             [
#                 "unemployed",
#                 "<1",
#                 "1<=X<4",
#                 "4<=X<7",
#                 ">=7"
#             ]
#         )

#         credit_history = st.selectbox(
#             "Credit History",
#             [
#                 "critical/other existing credit",
#                 "existing paid",
#                 "delayed previously",
#                 "no credits/all paid",
#                 "all paid"
#             ]
#         )

#     with col2:

#         purpose = st.selectbox(
#             "Purpose",
#             [
#                 "car",
#                 "furniture/equipment",
#                 "radio/tv",
#                 "domestic appliances",
#                 "repairs",
#                 "education",
#                 "vacation",
#                 "retraining",
#                 "business",
#                 "other"
#             ]
#         )

#         housing = st.selectbox(
#             "Housing",
#             [
#                 "own",
#                 "for free",
#                 "rent"
#             ]
#         )

#         job = st.selectbox(
#             "Job",
#             [
#                 "unemp/unskilled non res",
#                 "unskilled resident",
#                 "skilled",
#                 "high qualif/self emp/mgmt"
#             ]
#         )

#         personal_status_sex = st.selectbox(
#             "Personal Status / Sex",
#             [
#                 "male single",
#                 "female div/dep/mar",
#                 "male div/sep",
#                 "male mar/wid"
#             ]
#         )

#     st.divider()

#     # -----------------------------------------------------
#     # Additional categorical features
#     # -----------------------------------------------------

#     col1, col2 = st.columns(2)

#     with col1:

#         other_debtors = st.selectbox(
#             "Other Debtors",
#             [
#                 "none",
#                 "co-applicant",
#                 "guarantor"
#             ]
#         )

#         property_value = st.selectbox(
#             "Property",
#             [
#                 "real estate",
#                 "building society savings agreement/life insurance",
#                 "car or other",
#                 "unknown/no property"
#             ]
#         )

#         other_installment_plans = st.selectbox(
#             "Other Installment Plans",
#             [
#                 "none",
#                 "bank",
#                 "stores"
#             ]
#         )

#     with col2:

#         telephone = st.selectbox(
#             "Telephone",
#             [
#                 "none",
#                 "yes"
#             ]
#         )

#         foreign_worker = st.selectbox(
#             "Foreign Worker",
#             [
#                 "yes",
#                 "no"
#             ]
#         )

#         sex = st.selectbox(
#             "Sex",
#             [
#                 "male",
#                 "female"
#             ]
#         )

#     st.divider()

#     # =====================================================
#     # PREDICTION BUTTON
#     # =====================================================

#     if st.button(
#         "🔮 Predict Credit Risk",
#         use_container_width=True
#     ):

#         try:

#             # =================================================
#             # CREATE INPUT DATA
#             # =================================================

#             input_data = pd.DataFrame({
#                 "duration": [duration],
#                 "credit_amount": [credit_amount],
#                 "installment_rate": [installment_rate],
#                 "residence_duration": [residence_duration],
#                 "age": [age],
#                 "existing_credits": [existing_credits],
#                 "people_liable": [people_liable],
#                 "checking_account": [checking_account],
#                 "saving_account": [saving_account],
#                 "employment_duration": [employment_duration],
#                 "credit_history": [credit_history],
#                 "purpose": [purpose],
#                 "housing": [housing],
#                 "job": [job],
#                 "personal_status_sex": [personal_status_sex],
#                 "other_debtors": [other_debtors],
#                 "property": [property_value],
#                 "other_installment_plans": [
#                     other_installment_plans
#                 ],
#                 "telephone": [telephone],
#                 "foreign_worker": [foreign_worker]
#             })

#             # =================================================
#             # ALIGN FEATURE ORDER
#             # =================================================

#             if model_features is not None:

#                 missing_features = [
#                     feature
#                     for feature in model_features
#                     if feature not in input_data.columns
#                 ]

#                 if len(missing_features) > 0:

#                     st.error(
#                         "The following feature was not found :)"
#                         f"dalam input: {missing_features}"
#                     )

#                     st.stop()

#                 input_data = input_data[
#                     list(model_features)
#                 ]

#             # =================================================
#             # PREDICTION
#             # =================================================

#             prediction = (
#                 logistic_pipeline
#                 .predict(input_data)[0]
#             )

#             probabilities = (
#                 logistic_pipeline
#                 .predict_proba(input_data)[0]
#             )

#             classes = (
#                 logistic_pipeline.classes_
#             )

#             # =================================================
#             # GET PROBABILITY
#             # =================================================

#             good_index = list(
#                 classes
#             ).index("good")

#             bad_index = list(
#                 classes
#             ).index("bad")

#             probability_good = (
#                 probabilities[good_index]
#             )

#             probability_bad = (
#                 probabilities[bad_index]
#             )

#             # =================================================
#             # DISPLAY RESULT
#             # =================================================

#             st.subheader(
#                 "Prediction Result"
#             )

#             if prediction == "bad":

#                 st.error(
#                     "🔴 High Credit Risk — BAD"
#                 )

#             else:

#                 st.success(
#                     "🟢 Low Credit Risk — GOOD"
#                 )

#             # =================================================
#             # PROBABILITY
#             # =================================================

#             col1, col2 = st.columns(2)

#             with col1:

#                 st.metric(
#                     "Good Credit Probability",
#                     f"{probability_good:.2%}"
#                 )

#             with col2:

#                 st.metric(
#                     "Bad Credit Probability",
#                     f"{probability_bad:.2%}"
#                 )

#             st.divider()

#             # =================================================
#             # PROBABILITY BAR
#             # =================================================

#             st.subheader(
#                 "📊 Prediction Probability"
#             )

#             probability_df = pd.DataFrame({
#                 "Credit Risk": [
#                     "Good",
#                     "Bad"
#                 ],
#                 "Probability": [
#                     probability_good,
#                     probability_bad
#                 ]
#             })

#             st.bar_chart(
#                 probability_df.set_index(
#                     "Credit Risk"
#                 )
#             )

#         except Exception as e:

#             st.error(
#                 "The Prediction failed."
#             )

#             st.exception(e)