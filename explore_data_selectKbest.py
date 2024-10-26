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

# Step 4: Preprocessing - Handling Categorical Variables
# Identifying non-numeric columns
non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns

# Converting non-numeric columns to numeric using Label Encoding
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Splitting features and target variable
X = df.drop(
    ["loan_status", "id"], axis=1, errors="ignore"
)  # Assuming 'loan_status' is the target variable
y = (
    df["loan_status"] if "loan_status" in df.columns else None
)  # Setting target variable

# Create f_regression object to calculate F-value
f_value = f_classif(X, y)
print(f_value)

all_cols = df.columns

# Print the name and F-value of each feature
for feature in zip(all_cols, f_value[0]):
    print(feature)
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

    # Create a bar chart for visualizing the F-values
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_scores,
        x="Score",
        y="Feature",
        color="tomato",
    )
    plt.title("F-value of Features using SelectKBest (f_classif)", fontsize=16)
    plt.xlabel("F-value", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tight_layout()
    plt.show()

else:
    print(
        "The target variable 'loan_status' is not found in the dataset. Please check the column names."
    )
