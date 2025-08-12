# 🚢 Titanic Passenger Survival Prediction

## 📌 Overview
This project predicts whether a passenger survived the Titanic disaster using **machine learning**.  
It follows a complete end-to-end data science workflow:

1. **Data Analysis & Preprocessing**  
2. **Model Training & Evaluation**  
3. **Streamlit Web Application Development**  
4. **Model Deployment (optional)**  

The dataset used is the [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic), which contains demographic and travel information for Titanic passengers.

---

## 🎯 Objectives
- Perform **exploratory data analysis (EDA)** to understand the dataset.  
- Handle missing values and engineer meaningful features.  
- Train multiple classification models (Logistic Regression, Random Forest, SVM).  
- Evaluate models using **cross-validation** and select the best one.  
- Save the trained pipeline for future predictions.  
- Build a **Streamlit** app for interactive exploration and prediction.

---

## 🛠️ Technologies Used
- **Python** 3.9+  
- **Pandas** — Data manipulation  
- **NumPy** — Numerical computing  
- **Matplotlib / Seaborn / Plotly** — Visualisation  
- **Scikit-learn** — Machine learning models and pipelines  
- **Streamlit** — Web application  
- **Joblib** — Model saving/loading  

---

## 📂 Project Structure
titanic-project/
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
├── model.pkl # Trained ML pipeline
├── data/
│ ├── titanic.csv # Original dataset
│ └── test.csv # Saved test set for performance evaluation
├── notebooks/
│ └── model_training.ipynb # Jupyter Notebook (EDA, training, evaluation)
└── README.md # Project documentation


---

## 📊 Dataset Description
**Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  

| Column Name  | Description |
|--------------|-------------|
| Survived     | Target variable (0 = No, 1 = Yes) |
| Pclass       | Passenger class (1 = First, 2 = Second, 3 = Third) |
| Name         | Passenger name |
| Sex          | Gender |
| Age          | Age in years |
| SibSp        | Number of siblings/spouses aboard |
| Parch        | Number of parents/children aboard |
| Ticket       | Ticket number |
| Fare         | Ticket fare |
| Cabin        | Cabin number |
| Embarked     | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🧪 Workflow

### 1️⃣ Step 2 — Data Analysis & Preprocessing
- Checked data shape, columns, types, and missing values.
- Filled missing `Age`, `Fare`, and `Embarked` with median/mode values.
- Created new features:
  - `Title` (extracted from passenger names)
  - `FamilySize` and `IsAlone`
  - `HasCabin`
- Encoded categorical variables using **OneHotEncoder**.
- Split into **train** (80%) and **test** (20%) sets.

### 2️⃣ Step 3 — Model Training & Evaluation
- Models trained:  
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Used **cross-validation** for fair comparison.
- Evaluated with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Selected the **best-performing model** and saved it as `model.pkl` using Joblib.
- Saved the test set as `data/test.csv` for app performance display.

### 3️⃣ Step 4 — Streamlit Application
- **Navigation:** Sidebar with Home, Data Exploration, Visualisations, Model Prediction, Model Performance.
- **Data Exploration:** Shape, columns, data types, sample data, and filtering.
- **Visualisations:** Interactive charts for survival distribution, gender, class, age.
- **Prediction:** User inputs passenger details and receives prediction + survival probability.
- **Model Performance:** Classification report, confusion matrix, and model comparison table.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/titanic-project.git
cd titanic-project
