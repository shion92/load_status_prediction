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
import json
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

# Feature Engineering
grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["loan_grade_num"] = df["loan_grade"].map(grade_mapping)
df["grade_percent_income"] = df["loan_grade_num"] * df["loan_percent_income"]
df["grade_int_rate"] = df["loan_grade_num"] * df["loan_int_rate"]
df["percent_income_int_rate"] = df["loan_percent_income"] * df["loan_int_rate"]
df["loan_amnt_income_ratio"] = df["loan_amnt"] / df["person_income"]

# Standardizing specific features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    df[["loan_grade_num", "loan_percent_income", "loan_int_rate"]]
)
df[["loan_grade_scaled", "loan_percent_income_scaled", "loan_int_rate_scaled"]] = (
    scaled_features
)

df["loan_risk_score"] = (
    df["loan_grade_scaled"]
    + df["loan_percent_income_scaled"]
    + df["loan_int_rate_scaled"]
)

# Encoding, Binning, and Data Cleaning
df["emp_length_bin"] = pd.cut(
    df["person_emp_length"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["<2", "2-5", "5-10", "10-20", ">20"],
)

cb_default_mapping = {"Y": 1, "N": 0}
df["cb_person_default_on_file_num"] = df["cb_person_default_on_file"].map(
    cb_default_mapping
)

df["log_person_income"] = np.log1p(df["person_income"])
df["default_risk_1"] = df["cb_person_default_on_file_num"] * df["loan_int_rate"]
df["cred_length_grade"] = df["cb_person_cred_hist_length"] * df["loan_grade_num"]

# Removing Outliers
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
        "loan_amnt_income_ratio",
        "log_person_income",
        "cred_length_grade",
        "default_risk_1",
        "loan_amnt",
        "person_home_ownership",
    ]
]
# "loan_grade", "loan_percent_income",  "loan_int_rate",  , "cb_person_default_on_file",
y = df["loan_status"]

# Apply RandomUnderSampler to balance classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Log class distribution
log_message(
    f"\nClass distribution after undersampling:\n{pd.Series(y_resampled).value_counts()}"
)

# Standardize the Features
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Split the Resampled Data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)


# Define the DNN Model
def build_dnn_model(input_dim, num_layers, neurons, activation):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )
    return model


# Hyperparameters
num_layers_options = [2, 3, 5, 7, 9]
neurons = 5
epochs = 5000
batch_size = 128
activation_options = ["relu", "sigmoid"]
learning_rate = 0.0001
patience = 20

# Plotting Setup
fig, axes = plt.subplots(
    len(num_layers_options), len(activation_options), figsize=(20, 15)
)
fig.suptitle("Training Performance for Different DNN Configurations", fontsize=18)

best_accuracy = 0
best_config = None

# Model Training and Evaluation
for i, num_layers in enumerate(num_layers_options):
    for j, activation in enumerate(activation_options):
        log_message(
            f"\nTraining model with {num_layers} layers and '{activation}' activation (epochs = {epochs})..."
        )

        # Build the DNN model
        model = build_dnn_model(X_train.shape[1], num_layers, neurons, activation)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
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

        model_filename = f"model_{num_layers}layers_{activation}.h5"
        model_path = os.path.join("output_results/layers/models", model_filename)
        model.save(model_path)
        log_message(f"Model saved to: {model_path}")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        log_message(
            f"Accuracy: {accuracy * 100:.2f}% for {num_layers} layers, '{activation}' activation"
        )

        axes[i, j].plot(history.history["loss"], label="Training Loss")
        axes[i, j].plot(history.history["val_loss"], label="Validation Loss")
        axes[i, j].set_title(f"{num_layers} Layers, {activation} Activation")
        axes[i, j].set_xlabel("Epochs")
        axes[i, j].set_ylabel("Loss")
        axes[i, j].legend()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = (num_layers, activation)

# Save final plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
loss_accuracy_plot_file = "output_results/layers/plots/dnn_training_performance.png"
plt.savefig(loss_accuracy_plot_file)
plt.close()

# Log best configuration and accuracy
log_message(
    f"\nBest accuracy: {best_accuracy * 100:.2f}% with configuration: {best_config}"
)
log_message(f"Total script runtime: {time.time() - start_time:.2f} seconds")
