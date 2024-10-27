import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import matplotlib.pyplot as plt

# Load the preprocessed data (assuming the same scaling was used)
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# ---------- for all inputs model -----------
# Remove Outliers (Ensure columns exist before filtering)
df = df[
    (df["person_age"] <= 100)
    & (df["person_emp_length"] <= 60)
    & (df["person_income"] <= 1_000_000)
    & (df["loan_int_rate"] <= 35)
    & (df["cb_person_cred_hist_length"] <= 50)
]

# Encode Categorical Variables
non_numeric_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split Features and Target
X = df.drop(["loan_status", "id"], axis=1)
y = df["loan_status"]

# Apply RandomUnderSampler to balance the classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Standardize the Features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)


# # ---------- for subset model and multilayer -----------

# # Step 3: Feature Engineering
# # Mapping loan_grade to numerical values
# grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
# df["loan_grade_num"] = df["loan_grade"].map(grade_mapping)

# # Creating interaction and ratio features
# df["grade_percent_income"] = df["loan_grade_num"] * df["loan_percent_income"]  # -
# df["grade_int_rate"] = df["loan_grade_num"] * df["loan_int_rate"]
# df["percent_income_int_rate"] = df["loan_percent_income"] * df["loan_int_rate"]
# df["loan_amnt_income_ratio"] = df["loan_amnt"] / df["person_income"]

# # Standardizing specific features
# scaler = StandardScaler()
# df[["loan_grade_scaled", "loan_percent_income_scaled", "loan_int_rate_scaled"]] = (
#     scaler.fit_transform(df[["loan_grade_num", "loan_percent_income", "loan_int_rate"]])
# )

# # Creating a loan risk score
# df["loan_risk_score"] = (
#     df["loan_grade_scaled"]
#     + df["loan_percent_income_scaled"]
#     + df["loan_int_rate_scaled"]
# )

# # Binning age and employment length
# df["emp_length_bin"] = pd.cut(
#     df["person_emp_length"],
#     bins=[0, 2, 5, 10, 20, 50],
#     labels=["<2", "2-5", "5-10", "10-20", ">20"],
# )

# # Mapping default on file to numerical values
# cb_person_default_on_file_mapping = {"Y": 1, "N": 2}
# df["cb_person_default_on_file_mapping_num"] = df["cb_person_default_on_file"].map(
#     cb_person_default_on_file_mapping
# )

# # Log transformation of skewed features
# df["log_person_income"] = np.log1p(df["person_income"])

# # Creating additional risk-related features
# df["default_risk"] = df["cb_person_default_on_file_mapping_num"] * df["loan_int_rate"]

# df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]


# # Remove Outliers (Ensure columns exist before filtering)
# df = df[
#     (df["person_age"] <= 100)
#     & (df["person_emp_length"] <= 60)
#     & (df["person_income"] <= 1_000_000)
#     & (df["loan_int_rate"] <= 35)
#     & (df["cb_person_cred_hist_length"] <= 50)
# ]

# # Converting categorical bins to one-hot encoding
# df = pd.get_dummies(df, columns=["person_home_ownership"])
# print(df.head)

# # Encode Categorical Variables
# non_numeric_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
# label_encoders = {}
# for col in non_numeric_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# print(df.columns)

# X = df[
#     [
#         "grade_percent_income",
#         "loan_risk_score",
#         # "percent_income_int_rate",
#         # "grade_int_rate",
#         "loan_grade",
#         "loan_percent_income",
#         "loan_amnt_income_ratio",
#         "loan_int_rate",
#         "log_person_income",
#         "person_home_ownership_RENT",
#         "person_home_ownership_MORTGAGE",
#         # "cb_person_default_on_file",
#         # "default_risk",
#         # "loan_amnt",
#     ]
# ]
# print(len(X.columns))

# y = df["loan_status"]

# # Apply RandomUnderSampler to balance the classes in the target variable
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X, y)

# # Standardize the Features
# scaler = StandardScaler()
# X_resampled_scaled = scaler.fit_transform(X_resampled)

# Load the best saved model (update the file path as needed)
best_model_path = [
    "output_results/full/models/model_20.h5",  #  all data model
    # "output_results/subset/final_model_with_ratio_feature/models/model_15.h5",
    # "output_results/layers/models/model_2layers_relu.h5",
]

for model_path in best_model_path:
    print(model_path)
    model = load_model(model_path)

    # Define the Sobol analysis problem for the best-performing model
    problem = {
        "num_vars": X_resampled.shape[1],  # Number of input features
        "names": X.columns.tolist(),  # Feature names
        "bounds": [[0, 1]] * X_resampled.shape[1],  # Standardized feature range
    }

    # Generate Sobol samples using Saltelli sampling
    num_samples = 8192 * 4
    param_values = sobol.sample(problem, num_samples)

    # Scale the samples using the original scaler (use the same scaler applied during training)
    param_values_scaled = scaler.transform(param_values)

    # Run the model on the sampled inputs
    predictions = model.predict(param_values_scaled).flatten()

    # Sobol sensitivity analysis
    Si = sobol_analyze.analyze(problem, predictions, calc_second_order=True)

    # Display Sobol indices (first-order, second-order, and total-order)
    sensitivity_results = pd.DataFrame(
        {
            "Feature": problem["names"],
            "First-order": Si["S1"],
            "Total-order": Si["ST"],
        }
    )
    print("\n--- Sobol Sensitivity Analysis ---")
    print("First-order sensitivity indices (S1):")
    for name, s1, conf in zip(X.columns.tolist(), Si["S1"], Si["S1_conf"]):
        print(f"{name}: S1 = {s1:.4f} (±{conf:.4f})")

    print("\nTotal-order sensitivity indices (ST):")
    for name, st, conf in zip(X.columns.tolist(), Si["ST"], Si["ST_conf"]):
        print(f"{name}: ST = {st:.4f} (±{conf:.4f})")

    print("\nSecond-order sensitivity indices (S2):")
    for i, name_i in enumerate(X.columns.tolist()):
        for j, name_j in enumerate(X.columns.tolist()):
            if i < j:  # Print only the unique pairs
                s2_val = Si["S2"][i, j]
                conf_val = Si["S2_conf"][i, j]
                print(f"{name_i} and {name_j}: S2 = {s2_val:.4f} (±{conf_val:.4f})")

    # --- Additional Analysis: Visualization of Sensitivity Indices ---
    def plot_sensitivity_indices(Si, feature_names):
        """
        Plot S1 and ST sensitivity indices as bar plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ST (total-order indices)
        ax[0].barh(
            feature_names, Si["ST"], xerr=Si["ST_conf"], align="center", color="salmon"
        )
        ax[0].set_title("Total-order Sensitivity (ST)")
        ax[0].set_xlabel("ST")
        ax[0].set_xlim(0, 1)

        # Plot S1 (first-order indices)
        ax[1].barh(
            feature_names, Si["S1"], xerr=Si["S1_conf"], align="center", color="skyblue"
        )
        ax[1].set_title("First-order Sensitivity (S1)")
        ax[1].set_xlabel("S1")
        ax[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

    # Call the function to plot sensitivity indices
    plot_sensitivity_indices(Si, X.columns.tolist())
