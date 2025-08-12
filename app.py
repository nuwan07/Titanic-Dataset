import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Titanic Survival Prediction App", layout="wide")

# -------------------------------
# Load Data & Model
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./data/Titanic-Dataset.csv")  # Adjust path if needed

@st.cache_data
def load_test():
    return pd.read_csv("./data/test.csv")

@st.cache_resource
def load_model():
    return joblib.load("./notebooks/model.pkl")

df = load_data()
test_df = load_test()
model = load_model()

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
# Data Exploration Page
# -------------------------------
elif menu == "Data Exploration":
    st.header("📊 Data Exploration")

    st.subheader("Dataset Overview")

    st.dataframe(df.astype(str))
    
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns:**", list(df.columns))
    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Interactive Filter")
    col_choice = st.multiselect("Select columns to view:", df.columns, default=df.columns)
    rows_choice = st.slider("Number of rows to display:", 1, 50, 5)
    st.dataframe(df[col_choice].head(rows_choice))

# -------------------------------
# Visualisations Page
# -------------------------------
elif menu == "Visualisations":
    st.header("📈 Visualisations")

    # Chart 1: Survival Count
    st.subheader("Survival Count")
    fig1 = px.histogram(df, x="Survived", color="Survived", nbins=2)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Survival by Gender
    st.subheader("Survival by Gender")
    fig2 = px.histogram(df, x="Sex", color="Survived", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Age Distribution
    st.subheader("Age Distribution by Survival")
    fig3 = px.histogram(df, x="Age", color="Survived", nbins=30, marginal="box")
    st.plotly_chart(fig3, use_container_width=True)

    # Optional interactive filter
    st.subheader("Filter by Embarked Port")
    embarked_sel = st.selectbox("Select Embarked:", ["All"] + df["Embarked"].dropna().unique().tolist())
    if embarked_sel != "All":
        filtered_df = df[df["Embarked"] == embarked_sel]
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

    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

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
