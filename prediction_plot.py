import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import os

# Load the dataset
file_path = "data/train.csv"
df = pd.read_csv(file_path)

# Step 3: Feature Engineering
# Mapping loan_grade to numerical values
grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["loan_grade_num"] = df["loan_grade"].map(grade_mapping)

# Creating interaction and ratio features
df["grade_percent_income"] = df["loan_grade_num"] * df["loan_percent_income"]  # -
df["grade_int_rate"] = df["loan_grade_num"] * df["loan_int_rate"]
df["percent_income_int_rate"] = df["loan_percent_income"] * df["loan_int_rate"]
df["loan_amnt_income_ratio"] = df["loan_amnt"] / df["person_income"]

# Standardizing specific features
scaler = StandardScaler()
df[["loan_grade_scaled", "loan_percent_income_scaled", "loan_int_rate_scaled"]] = (
    scaler.fit_transform(df[["loan_grade_num", "loan_percent_income", "loan_int_rate"]])
)

# Creating a loan risk score
df["loan_risk_score"] = (
    df["loan_grade_scaled"]
    + df["loan_percent_income_scaled"]
    + df["loan_int_rate_scaled"]
)

# Binning age and employment length
df["emp_length_bin"] = pd.cut(
    df["person_emp_length"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["<2", "2-5", "5-10", "10-20", ">20"],
)

# Mapping default on file to numerical values
cb_person_default_on_file_mapping = {"Y": 1, "N": 2}
df["cb_person_default_on_file_mapping_num"] = df["cb_person_default_on_file"].map(
    cb_person_default_on_file_mapping
)

# Log transformation of skewed features
df["log_person_income"] = np.log1p(df["person_income"])

# Creating additional risk-related features
df["default_risk"] = df["cb_person_default_on_file_mapping_num"] * df["loan_int_rate"]

df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]


# Remove Outliers (Ensure columns exist before filtering)
df = df[
    (df["person_age"] <= 100)
    & (df["person_emp_length"] <= 60)
    & (df["person_income"] <= 1_000_000)
    & (df["loan_int_rate"] <= 35)
    & (df["cb_person_cred_hist_length"] <= 50)
]

# Converting categorical bins to one-hot encoding
df = pd.get_dummies(df, columns=["person_home_ownership"])
print(df.head)

# Encode Categorical Variables
non_numeric_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print(df.columns)

X = df[
    [
        "grade_percent_income",
        "loan_risk_score",
        # "percent_income_int_rate",
        # "grade_int_rate",
        "loan_grade",
        "loan_percent_income",
        "loan_amnt_income_ratio",
        "loan_int_rate",
        "log_person_income",
        "person_home_ownership_RENT",
        "person_home_ownership_MORTGAGE",
        # "cb_person_default_on_file",
        # "default_risk",
        # "loan_amnt",
    ]
]
print(len(X.columns))

y = df["loan_status"]

# Apply RandomUnderSampler to balance the classes in the target variable
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Standardize the Features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Split the Resampled Data into Training and Testing Sets
# splitting ratio train : test : validation ==> 0.6 : 0.2 : 0.2
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)
# Load the best model from the saved path
saved_model_path = (
    "output_results/subset/final_model_with_ratio_feature/models/model_15.h5"
)
model = load_model(saved_model_path)

# Make Predictions and Classification Report on test data
predicted_y_test = model.predict(X_test)
predicted_test = (predicted_y_test >= 0.5).astype(int).flatten()

classification_rep_test = classification_report(y_test, predicted_test)
print("classification_rep_test: ", classification_rep_test)
# Display Confusion Matrix for test data
conf_mat_test = confusion_matrix(y_test, predicted_test)
# Confusion Matrix Plot for test data
fig_conf, ax_conf = plt.subplots(1, 2, figsize=(10, 4))
fig_conf.suptitle(f"Confusion Matrix for 15 Neurons on Test and All Data", fontsize=14)

ConfusionMatrixDisplay.from_predictions(
    y_test, predicted_test, ax=ax_conf[0], cmap="Blues"
)
ax_conf[0].set_title("Test")

# Make Predictions and Classification Report on ALL data
predicted_y_all = model.predict(X_resampled_scaled)
predicted_all = (predicted_y_all >= 0.5).astype(int).flatten()

classification_rep_all = classification_report(y_resampled, predicted_all)
print("classification_rep_all:", classification_rep_all)

# Display Confusion Matrix for ALL data
conf_mat_all = confusion_matrix(y_resampled, predicted_all)

ConfusionMatrixDisplay.from_predictions(
    y_resampled, predicted_all, ax=ax_conf[1], cmap="Blues"
)

conf_matrix_file = os.path.join(
    "output_results/subset/plots", f"conf_matrix_all_and_test_neuron15.png"
)
fig_conf.savefig(conf_matrix_file)
plt.close(fig_conf)
ax_conf[0].set_title("ALL")
