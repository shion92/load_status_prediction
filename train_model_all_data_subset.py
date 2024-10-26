# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os
import time
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.callbacks import EarlyStopping

# Start measuring the total runtime
start_time = time.time()

# Create directories for saving outputs
os.makedirs("output_results/layers/plots", exist_ok=True)
os.makedirs("output_results/layers/models", exist_ok=True)
output_log = "output_results/layers/non_torch_output_log.txt"


# Function to write to output log
def log_message(message):
    with open(output_log, "a") as f:
        f.write(message + "\n")
    print(message)


# Load and Preprocess the Data
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)


# Step 3: Feature Engineering (remains the same as provided)
# Mapping loan_grade to numerical values
grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["loan_grade_num"] = df["loan_grade"].map(grade_mapping)

# Creating interaction and ratio features
df["grade_percent_income"] = df["loan_grade_num"] * df["loan_percent_income"]
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
df["default_risk_1"] = df["cb_person_default_on_file_mapping_num"] * df["loan_int_rate"]
df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]

# Remove Outliers (Ensure columns exist before filtering)
df = df[
    (df["person_age"] <= 100)
    & (df["person_emp_length"] <= 60)
    & (df["person_income"] <= 1_000_000)
    & (df["loan_int_rate"] <= 35)
    & (df["cb_person_cred_hist_length"] <= 50)
]

# Encode Categorical Variables
non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split Features and Target
X = df[
    [
        "grade_percent_income",
        "loan_risk_score",
        "percent_income_int_rate",
        "grade_int_rate",
        "loan_grade",
        "loan_percent_income",
        "loan_amnt_income_ratio",
        "loan_int_rate",
        "log_person_income",
        "person_home_ownership",
        "cred_length_grade",
        "cb_person_default_on_file",
        "default_risk_1",
        "loan_amnt",
    ]
]
y = df["loan_status"]

# Apply RandomUnderSampler to balance the classes in the target variable
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Log the class distribution after undersampling
log_message("\nClass distribution after undersampling:")
log_message(str(pd.Series(y_resampled).value_counts()))

# Standardize the Features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Split the Resampled Data into Training and Testing Validation Sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)


# Define the DNN Model with variable number of layers and neurons
def build_dnn_model(input_dim, num_layers, neurons, activation):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation=activation))

    for _ in range(num_layers - 1):
        model.add(Dense(neurons, activation=activation))

    model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )
    return model


# Hyperparameters
num_layers_options = [1, 2, 3, 4]  # Testing 1 to 4 hidden layers
neurons = 5  # Fixed number of neurons in each layer
epochs = 2000  # Reduced epochs to prevent overfitting
batch_size = 128
activation_options = ["relu", "sigmoid"]  # Activation functions to test

# Plotting Setup
fig, axes = plt.subplots(
    len(num_layers_options), len(activation_options), figsize=(12, 8)
)
fig.suptitle("Training Performance for Different DNN Configurations", fontsize=16)

best_accuracy = 0
best_config = None

for i, num_layers in enumerate(num_layers_options):
    for j, activation in enumerate(activation_options):
        log_message(
            f"\nTraining model with {num_layers} layers and '{activation}' activation..."
        )

        # Build the DNN model
        model = build_dnn_model(X_train.shape[1], num_layers, neurons, activation)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Evaluate the model on test data
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        log_message(
            f"Accuracy with {num_layers} layers, '{activation}' activation: {accuracy * 100:.2f}%"
        )

        # Plot Loss and Accuracy
        axes[i, j].plot(history.history["loss"], label="Training Loss")
        axes[i, j].plot(history.history["val_loss"], label="Validation Loss")
        axes[i, j].set_title(f"{num_layers} Layers, {activation} Activation")
        axes[i, j].set_xlabel("Epochs")
        axes[i, j].set_ylabel("Loss")
        axes[i, j].legend()

        # Update best accuracy and configuration
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = (num_layers, activation)

        # Classification report on test data
        predicted_y_test = model.predict(X_test)
        predicted_test = (predicted_y_test >= 0.5).astype(int).flatten()

        classification_rep_test = classification_report(y_test, predicted_test)
        log_message(f"\nClassification report:\n{classification_rep_test}")

# Save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("output_results/layers/plots/dnn_training_performance.png")
plt.show()

# Log the best configuration and accuracy
log_message(
    f"\nBest accuracy achieved: {best_accuracy * 100:.2f}% with configuration: {best_config}"
)
log_message(f"Total script runtime: {time.time() - start_time:.2f} seconds")
