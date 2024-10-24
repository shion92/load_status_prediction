# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)


# Selecting numerical features only
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Step 7: Plotting Histograms of Numerical Features
df[numerical_cols].hist(bins=20, figsize=(15, 12), edgecolor="black")
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.show()

# Step 3: Remove Outliers
# Applying outlier removal logic based on provided criteria and common sense
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

# Step 4: Handling Categorical Variables
# Identifying non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object"]).columns

# Step 5: Visualizing Numerical Features with Box Plots by 'loan_status'
plt.figure(figsize=(18, 10))


for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 4, i)  # Adjusting the layout to accommodate all plots
    sns.boxplot(x="loan_status", y=col, data=df)
    plt.title(f"Box Plot of {col} by Loan Status")

plt.tight_layout()
plt.show()

# Step 6: Correlation Matrix and Heatmap for numerical features
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# Number of non-numeric columns
num_cols = len(non_numeric_cols)

# Step 8a: Visualizing Non-Numeric Features with Stacked Bar Plots (Count) by 'loan_status'
fig, axes = plt.subplots(
    num_cols, 2, figsize=(14, 6 * num_cols), constrained_layout=True
)

for i, col in enumerate(non_numeric_cols):
    # Grouping data by 'loan_status' and the categorical feature
    grouped_data = df.groupby([col, "loan_status"]).size().unstack()

    # Plotting stacked bar plot for count distribution in the first column of subplots
    grouped_data.plot(kind="bar", stacked=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"{col} (Count)")
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel("Count")

    # Step 8b: Visualizing Non-Numeric Features with Stacked Bar Plots (Percentage) by 'loan_status'
    # Converting counts to percentages
    grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

    # Plotting stacked bar plot for percentage distribution in the second column of subplots
    grouped_data_percentage.plot(
        kind="bar", stacked=True, ax=axes[i, 1], colormap="viridis"
    )
    axes[i, 1].set_title(f"{col} (Percentage)")
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel("Percentage")

# Setting a single overarching title
fig.suptitle("Stacked Bar Plots of Non-Numeric Features by Loan Status", fontsize=18)
plt.show()
