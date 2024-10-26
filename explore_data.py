# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)
df = df.drop("id", axis=1)
print(df.info())
print(df.describe().T)
print(df.nunique())

# Selecting numerical features only
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_cols:
    print(f"Column: {col}")
    print(f"Unique Values: {df[col].unique()}")
    print("\n")

# Identifying non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object"]).columns
for col in non_numeric_cols:
    print(f"Column: {col}")
    print(f"Unique Values: {df[col].unique()}")
    print("\n")


# # Step 7: Plotting Histograms of Numerical Features
# df[numerical_cols].hist(bins=20, figsize=(15, 12), edgecolor="black")
# plt.suptitle("Histograms of Numerical Features", y=1.02)
# plt.show()

# # Step 3: Remove Outliers
# # Applying outlier removal logic based on provided criteria and common sense
# df = df[
#     (df["person_age"] <= 100)  # Remove records with 'person_age' greater than 100
#     & (
#         df["person_emp_length"] <= 60
#     )  # Remove records with 'person_emp_length' greater than 60
#     & (df["person_income"] <= 1_000_000)  # Assuming reasonable income limit
#     & (df["loan_int_rate"] <= 35)  # Assuming max interest rate of 35%
#     & (
#         df["cb_person_cred_hist_length"] <= 50
#     )  # Assuming max credit history length of 50 years
# ]

# # Step 4: Bivariate Analysis
# fig, axes = plt.subplots(4, 2, figsize=(16, 20))
# fig.suptitle("Numerical Features vs Loan Status (Density Plots)", fontsize=16)

# for i, col in enumerate(numerical_cols):
#     sns.kdeplot(
#         data=df,
#         x=col,
#         hue="loan_status",
#         ax=axes[i // 2, i % 2],
#         fill=True,
#         common_norm=False,
#         palette="Set2",
#     )
#     axes[i // 2, i % 2].set_title(f"{col} vs Loan Status")
#     axes[i // 2, i % 2].set_xlabel(col)
#     axes[i // 2, i % 2].set_ylabel("Density")

# fig.delaxes(axes[3, 1])

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# # Step 5: Visualizing Numerical Features with Box Plots by 'loan_status

# # Apply log transformation to the specified numerical columns
loan_status = df["loan_status"]

# # Apply log transformation to the specified numerical columns
# df_log_transformed = df[numerical_cols].copy()

# # Selecting only the numerical columns, excluding 'loan_status'
# numerical_cols = df_log_transformed.select_dtypes(include=["number"]).columns.drop(
#     "loan_status"
# )
# df_log_transformed = np.log1p(df_log_transformed)

# # Add 'loan_status' back to the log-transformed DataFrame
# df_log_transformed["loan_status"] = loan_status

# # Plotting box plots for each column
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(numerical_cols, 1):
#     plt.subplot(2, 4, i)  # Adjusting the layout to accommodate all plots
#     sns.boxplot(x="loan_status", y=col, data=df_log_transformed)
#     # plt.title(f"Box Plot of {col} by Loan Status (Log scale)")
# plt.show()

# # Step 6: Correlation Matrix and Heatmap for numerical features
# # df["loan_status"] = loan_status
# numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# corr_matrix = df[numerical_cols].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Heatmap of Numerical Features")
# plt.show()


# # Number of non-numeric columns
# num_cols = len(non_numeric_cols)


# # Function to wrap long labels
# def wrap_label(label, width=10):
#     return "\n".join(textwrap.wrap(label, width))


# # Step 8a: Visualizing Non-Numeric Features with Stacked Bar Plots (Count) by 'loan_status'
# fig, axes = plt.subplots(num_cols, 2, figsize=(14, 20), constrained_layout=True)

# for i, col in enumerate(non_numeric_cols):
#     # Grouping data by 'loan_status' and the categorical feature
#     grouped_data = df.groupby([col, "loan_status"]).size().unstack()

#     # Plotting stacked bar plot for count distribution in the first column of subplots
#     grouped_data.plot(kind="bar", stacked=True, ax=axes[i, 0])
#     axes[i, 0].set_title(f"{col} (Count)")
#     axes[i, 0].set_xlabel(col)
#     axes[i, 0].set_ylabel("Count")
#     axes[i, 0].tick_params(axis="x", rotation=0)

#     # Step 8b: Visualizing Non-Numeric Features with Stacked Bar Plots (Percentage) by 'loan_status'
#     # Converting counts to percentages
#     grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

#     # Plotting stacked bar plot for percentage distribution in the second column of subplots
#     grouped_data_percentage.plot(
#         kind="bar", stacked=True, ax=axes[i, 1], colormap="viridis"
#     )
#     axes[i, 1].set_title(f"{col} (Percentage)")
#     axes[i, 1].set_xlabel(col)
#     axes[i, 1].set_ylabel("Percentage")
#     axes[i, 1].tick_params(axis="x", rotation=0)

# # Setting a single overarching title
# fig.suptitle("Stacked Bar Plots of Non-Numeric Features by Loan Status", fontsize=18)
# plt.show()


# Apply Label Encoding to 'loan_grade' and 'cb_person_default_on_file'
label_encoder = LabelEncoder()

# Label Encode 'loan_grade'
df["loan_grade"] = label_encoder.fit_transform(df["loan_grade"])

# Label Encode 'cb_person_default_on_file'
df["cb_person_default_on_file"] = label_encoder.fit_transform(
    df["cb_person_default_on_file"]
)

# Apply One-Hot Encoding to 'person_home_ownership' and 'loan_intent'
df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"])

print(df.columns)

# Step 6: Correlation Matrix and Heatmap for numerical features
# df["loan_status"] = loan_status
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

corr_matrix = df[
    [
        "loan_grade",
        "cb_person_default_on_file",
        "loan_status",
        "person_home_ownership_MORTGAGE",
        "person_home_ownership_OTHER",
        "person_home_ownership_OWN",
        "person_home_ownership_RENT",
        "loan_intent_DEBTCONSOLIDATION",
        "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT",
        "loan_intent_MEDICAL",
        "loan_intent_PERSONAL",
        "loan_intent_VENTURE",
    ]
].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Non-Numerical Features")
plt.show()
