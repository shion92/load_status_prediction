# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
file_path = "data/train.csv"  # Update the path if necessary
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
# df["age_bin"] = pd.cut(
#     df["person_age"],
#     bins=[0, 15, 25, 35, 45, 55, 100],
#     labels=["<15", "15-25", "25-35", "35-45", "45-55", ">55"],
# )

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
df["default_risk_1"] = df["cb_person_default_on_file_mapping_num"] * df["loan_int_rate"]

df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]


# Step 4: Remove Outliers
df = df[
    (df["person_age"] <= 100)  # Remove records with 'person_age' greater than 100
    & (
        df["person_emp_length"] <= 60
    )  # Remove records with 'person_emp_length' greater than 60
    & (df["person_income"] <= 1_000_000)  # Assuming reasonable income limit
    & (df["loan_int_rate"] <= 35)  # Assuming max interest rate of 35%
    & (
        df["cb_person_cred_hist_length"] <= 50
    )  # Assuming max credit history length of 50 years
]

df = df.drop(
    [
        "id",
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_intent",
        "cb_person_cred_hist_length",
        "loan_grade_scaled",
        "loan_grade_num",
        "loan_grade_scaled",
        "loan_int_rate_scaled",
        "cb_person_default_on_file_mapping_num",
        # "age_bin",
    ],
    axis=1,
)

print(df.dtypes)

# Step 5: Handling Categorical Variables
non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns
print(non_numeric_cols)

# Label Encoding non-numeric columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# # Converting categorical bins to one-hot encoding
# df = pd.get_dummies(df, columns=["age_bin", "emp_length_bin"], drop_first=True)

# Step 6: Selecting Numerical Features
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Step 7: Plotting Histograms of Numerical Features
df[numerical_cols].hist(bins=20, figsize=(15, 12), edgecolor="black")
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.show()

# Step 8: Visualizing Numerical Features with Box Plots by 'loan_status'
plt.figure(figsize=(18, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 5, i)  # Adjusting the layout to accommodate all plots
    sns.boxplot(x="loan_status", y=col, data=df)
    plt.title(f"Box Plot of {col} by Loan Status")

plt.tight_layout()
plt.show()

# Step 9: Correlation Matrix and Heatmap for Numerical Features
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Step 10: Visualizing Non-Numeric Features with Stacked Bar Plots by 'loan_status'
num_cols = len(non_numeric_cols)
fig, axes = plt.subplots(
    num_cols, 2, figsize=(14, 6 * num_cols), constrained_layout=True
)

for i, col in enumerate(non_numeric_cols):
    # Grouping data by 'loan_status' and the categorical feature
    grouped_data = df.groupby([col, "loan_status"]).size().unstack()

    # Plotting stacked bar plot for count distribution
    grouped_data.plot(kind="bar", stacked=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"{col} (Count)")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel("Count")

    # Plotting stacked bar plot for percentage distribution
    grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100
    grouped_data_percentage.plot(
        kind="bar", stacked=True, ax=axes[i, 1], colormap="viridis"
    )
    axes[i, 1].set_title(f"{col} (Percentage)")
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel("Percentage")

# Setting a single overarching title
fig.suptitle("Stacked Bar Plots of Non-Numeric Features by Loan Status", fontsize=18)
plt.show()
