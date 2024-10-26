# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# Step 3: Exploratory Data Analysis (EDA)
# 3.1: Basic statistical summary of the dataset
df_summary = df.describe()
print(df_summary)

# print(df.dtypes)

grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["loan_grade_num"] = df["loan_grade"].map(grade_mapping)

df["grade_percent_income"] = df["loan_grade_num"] * df["loan_percent_income"]
df["grade_int_rate"] = df["loan_grade_num"] * df["loan_int_rate"]
df["percent_income_int_rate"] = df["loan_percent_income"] * df["loan_int_rate"]
df["loan_amnt_income_ratio"] = df["loan_amnt"] / df["person_income"]


scaler = StandardScaler()
df[["loan_grade_scaled", "loan_percent_income_scaled", "loan_int_rate_scaled"]] = (
    scaler.fit_transform(df[["loan_grade_num", "loan_percent_income", "loan_int_rate"]])
)
df["loan_risk_score"] = (
    df["loan_grade_scaled"]
    + df["loan_percent_income_scaled"]
    + df["loan_int_rate_scaled"]
)
df["age_bin"] = pd.cut(
    df["person_age"],
    bins=[0, 15, 25, 35, 45, 55, 100],
    labels=["<15", "15-25", "25-35", "35-45", "45-55", ">55"],
)

df["emp_length_bin"] = pd.cut(
    df["person_emp_length"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["<2", "2-5", "5-10", "10-20", ">20"],
)

df["high_int_rate"] = (df["loan_int_rate"] > 0.15).astype(
    int
)  # Assuming 15% as the threshold

cb_person_default_on_file_mapping = {"Y": 1, "N": 2}
df["cb_person_default_on_file_mapping_num"] = df["cb_person_default_on_file"].map(
    cb_person_default_on_file_mapping
)


df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
df["log_person_income"] = np.log1p(df["person_income"])

df["default_risk_1"] = df["cb_person_default_on_file_mapping_num"] * df["loan_int_rate"]
# df["default_risk_2"] = df["person_age"] / df["person_emp_length"] * df["log_loan_amnt"]

df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]


# Step 4: Preprocessing - Handling Categorical Variables
# Identifying non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns

# Converting non-numeric columns to numeric using Label Encoding
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(df.head)

# # Convert categorical columns to numerical using one-hot encoding
# df = pd.get_dummies(df, columns=["age_bin", "emp_length_bin"], drop_first=True)

# Step 5: Splitting features and target variable
X = df.drop(
    [
        "loan_status",
        "id",
        "person_age",
        "person_emp_length",
        "loan_intent",
        "cb_person_cred_hist_length",
        "loan_grade_scaled",
        "loan_grade_num",
    ],
    axis=1,
)

X = X.select_dtypes(include=[np.number])  # Select only numeric columns
X.head(10)
y = df["loan_status"]  # Setting target variable


# print(np.isinf(X).sum())  # Check for infinite values
# print(np.isnan(X).sum())  # Check for NaN values

# Step 6: Standardizing the features
if y is not None:  # Proceed only if the target variable is available
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 7: Using SelectKBest with f_classif to determine top features
    select_k_best = SelectKBest(score_func=f_classif, k="all")
    fit = select_k_best.fit(X_scaled, y)

    # Step 8: Creating a DataFrame to visualize feature scores
    feature_scores = pd.DataFrame(
        {"Feature": X.columns, "Score": fit.scores_}
    ).sort_values(by="Score", ascending=False)

    # Displaying the feature scores
    print("Feature Scores from SelectKBest (using f_classif):")
    print(feature_scores)
else:
    print(
        "The target variable 'loan_status' is not found in the dataset. Please check the column names."
    )
