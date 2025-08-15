import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- ensure _arrow_safe is defined before loaders so loaders can return safe dfs ---
def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    # drop accidental unnamed columns from CSV exports
    drop_cols = [c for c in out.columns if (isinstance(c, str) and (c.strip() == "" or c.startswith("Unnamed")))]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    for col in out.columns:
        ser = out[col]

        # integer-like (includes pandas nullable "Int64")
        if pd.api.types.is_integer_dtype(ser) or str(ser.dtype).startswith("Int"):
            out[col] = ser.astype("float64").to_numpy() if ser.isna().any() else ser.astype("int64").to_numpy()
            continue

        # floats
        if pd.api.types.is_float_dtype(ser):
            out[col] = ser.astype("float64").to_numpy()
            continue

        # booleans (incl. pandas nullable "boolean")
        if pd.api.types.is_bool_dtype(ser) or str(ser.dtype) == "boolean":
            out[col] = ser.fillna(False).astype("bool").to_numpy()
            continue

        # datetimes
        if pd.api.types.is_datetime64_any_dtype(ser):
            out[col] = pd.to_datetime(ser, errors="coerce").to_numpy()
            continue

        # categorical
        if isinstance(ser.dtype, pd.CategoricalDtype):
            out[col] = ser.astype(str).fillna("").to_numpy()
            continue

        # fallback: object/mixed -> string
        out[col] = ser.fillna("").astype(str).to_numpy()

    return pd.DataFrame({c: out[c] for c in out.columns})

# -------------------------------
# Load Data & Model (return Arrow-safe dfs)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"./data/Titanic-Dataset.csv", low_memory=False)
    # drop accidental index/unnamed column(s)
    drop_cols = [c for c in df.columns if isinstance(c, str) and (c.strip() == "" or c.startswith("Unnamed"))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return _arrow_safe(df)

@st.cache_data
def load_test():
    df = pd.read_csv(r"./data/test.csv", low_memory=False)
    drop_cols = [c for c in df.columns if isinstance(c, str) and (c.strip() == "" or c.startswith("Unnamed"))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return _arrow_safe(df)

@st.cache_resource
def load_model():
    return joblib.load(r"./notebooks/model.pkl")

df = load_data()
test_df = load_test()
model = load_model()

# convert datasets to Arrow-safe forms immediately after loading
df_ui = _arrow_safe(df)
test_df_ui = _arrow_safe(test_df)


# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", 
    ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# -------------------------------
# Home Page
# -------------------------------
if menu == "Home":
    st.title("🚢 Titanic Passenger Survival Prediction")
    st.markdown("""
    ### Project Description
    This application predicts whether a passenger would have survived the Titanic disaster
    based on personal and travel details.  
    **Features**:
    - Data exploration with filtering
    - Interactive visualisations
    - Real-time prediction with confidence score
    - Model performance metrics and comparison
    """)
    st.info("Use the sidebar to navigate between sections.")

# -------------------------------
# Data Exploration Page (removed)
# -------------------------------
elif menu == "Data Exploration":
    st.header("📊 Data Exploration")
    st.info("Data Exploration section removed to avoid Arrow serialization issues. Use Visualisations or Model pages.")

# -------------------------------
# Visualisations Page
# -------------------------------
elif menu == "Visualisations":
    st.header("📈 Visualisations")

    # Chart 1: Survival Count
    st.subheader("Survival Count")
    fig1 = px.histogram(df_ui, x="Survived", color="Survived", nbins=2)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Survival by Gender
    st.subheader("Survival by Gender")
    fig2 = px.histogram(df_ui, x="Sex", color="Survived", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Age Distribution
    st.subheader("Age Distribution by Survival")
    fig3 = px.histogram(df_ui, x="Age", color="Survived", nbins=30, marginal="box")
    st.plotly_chart(fig3, use_container_width=True)

    # Optional interactive filter
    st.subheader("Filter by Embarked Port")
    embarked_sel = st.selectbox("Select Embarked:", ["All"] + df_ui["Embarked"].dropna().unique().tolist())
    if embarked_sel != "All":
        filtered_df = df_ui[df_ui["Embarked"] == embarked_sel]
        fig4 = px.histogram(filtered_df, x="Pclass", color="Survived", barmode="group")
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# Model Prediction Page
# -------------------------------
elif menu == "Model Prediction":
    st.header("🤖 Model Prediction")

    st.write("Enter passenger details to predict survival:")

    with st.form("prediction_form"):
        pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, value=32.0)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"], index=0)
        family_size = st.number_input("FamilySize", min_value=1, max_value=20, value=1)
        is_alone = 1 if family_size == 1 else 0
        has_cabin = st.checkbox("Has Cabin", value=False)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            X_new = pd.DataFrame([{
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
                "Title": title,
                "FamilySize": family_size,
                "IsAlone": is_alone,
                "HasCabin": int(has_cabin)
            }])

            pred = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0][1]

            st.success(f"Prediction: {'Survived' if pred == 1 else 'Did not survive'}")
            st.info(f"Prediction Confidence: {proba:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Model Performance Page
# -------------------------------
elif menu == "Model Performance":
    st.header("📊 Model Performance")

    X_test = test_df_ui.drop(columns=["Survived"])
    y_test = test_df_ui["Survived"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Model Comparison (CV Results)")
    st.write("Comparison table from training (manually enter from your notebook):")
    # Example placeholder table - replace with your actual results_df
    example_results = pd.DataFrame({
        "accuracy": [0.82, 0.85, 0.78],
        "precision": [0.80, 0.86, 0.77],
        "recall": [0.74, 0.82, 0.75],
        "f1": [0.77, 0.84, 0.76],
        "roc_auc": [0.88, 0.90, 0.85]
    }, index=["Logistic Regression", "Random Forest", "SVM"])
    st.dataframe(example_results)
