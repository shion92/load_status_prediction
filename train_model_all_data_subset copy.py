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
from tensorflow import squeeze
import matplotlib.pyplot as plt
import time
import os
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.callbacks import EarlyStopping

# Start measuring the total runtime
start_time = time.time()

# Create directories for saving outputs
os.makedirs("output_results/subset/plots", exist_ok=True)
os.makedirs("output_results/subset/models", exist_ok=True)
output_log = "output_results/subset/non_torch_output_log.txt"


# Function to write to output log
def log_message(message):
    with open(output_log, "a") as f:
        f.write(message + "\n")
    print(message)


# Load and Preprocess the Data
file_path = "data/train.csv"  # Update the path if necessary
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    log_message(f"File not found: {file_path}")
    raise

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
# X = df.drop(
#     [
#         "loan_status",
#         "id",
#         "person_age",
#         "person_emp_length",
#         "loan_intent",
#         "cb_person_cred_hist_length",
#         "person_income",
#         "loan_grade_scaled",
#         "loan_grade_num",
#         "loan_grade_scaled",
#         "loan_int_rate_scaled",
#         "cb_person_default_on_file_mapping_num",
#     ],
#     axis=1,
# )

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
print(len(X.columns))

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

# Split the Resampled Data into Training and Testing Sets
# splitting ratio train : test : validation ==> 0.6 : 0.2 : 0.2
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)


# Define the MLP Model with One Hidden Layer and Dropout
def build_model(hidden_neurons):
    model = Sequential(
        [
            Dense(
                hidden_neurons,
                input_dim=X_train.shape[1],
                activation="sigmoid",
            ),
            # Dropout(0.3),  # Dropout layer to prevent overfitting
            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )
    return model


# Hyperparameters
best_accuracy = 0
best_neurons = 0
neuron_options = [5, 10, 15, 20]
epochs = 2000  # Reduced epochs to prevent overfitting
learning_rate = 0.001
batch_size = 128

fig, axes = plt.subplots(len(neuron_options), 2, figsize=(14, 4 * len(neuron_options)))
fig.suptitle("Training Performance for Different Neurons", fontsize=16)

for i, neurons in enumerate(neuron_options):
    log_message(
        f"\nTraining model with {neurons} neurons in the hidden layer for {epochs} epochs..."
    )

    # Start timing for the current model
    start_model_time = time.time()
    model = build_model(neurons)

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

    # Save the trained model
    model_file = os.path.join("output_results/subset/models", f"model_{neurons}.h5")
    model.save(model_file)
    log_message(f"Model saved to: {model_file}")

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log_message(f"Accuracy with {neurons} neurons on test data: {accuracy * 100:.2f}%")
    log_message(
        f"Time taken for {neurons} neurons: {time.time() - start_model_time:.2f} seconds"
    )

    # Plot Loss
    axes[i, 0].plot(history.history["loss"], label="Training Loss")
    axes[i, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[i, 0].set_title(f"Loss for {neurons} Neurons")
    axes[i, 0].set_xlabel("Epochs")
    axes[i, 0].set_ylabel("Loss")
    axes[i, 0].legend()

    # Plot Accuracy
    axes[i, 1].plot(history.history["binary_accuracy"], label="Training Accuracy")
    axes[i, 1].plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    axes[i, 1].set_title(f"Accuracy for {neurons} Neurons")
    axes[i, 1].set_xlabel("Epochs")
    axes[i, 1].set_ylabel("Accuracy")
    axes[i, 1].legend()

    # Update best accuracy and neurons
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_neurons = neurons

    # Make Predictions and Classification Report on test data
    predicted_y_test = model.predict(X_test)
    predicted_test = (predicted_y_test >= 0.5).astype(int).flatten()

    classification_rep_test = classification_report(y_test, predicted_test)
    log_message(
        f"\nClassification report for {neurons} neurons on test data:\n{classification_rep_test}"
    )

    # Display Confusion Matrix for test data
    conf_mat_test = confusion_matrix(y_test, predicted_test)
    log_message(
        f"Confusion Matrix for {neurons} neurons on test data:\n{conf_mat_test}"
    )

    # Confusion Matrix Plot for test data
    fig_conf_test, ax_conf_test = plt.subplots(1, 2, figsize=(10, 4))
    fig_conf_test.suptitle(
        f"Confusion Matrix for {neurons} Neurons on Test Data", fontsize=14
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted_test, ax=ax_conf_test[0], cmap="Blues"
    )
    ax_conf_test[0].set_title("Confusion Matrix (Raw)")

    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted_test, normalize="true", ax=ax_conf_test[1], cmap="Blues"
    )
    ax_conf_test[1].set_title("Confusion Matrix (Normalized)")

    # Save Confusion Matrix Plot for test data
    conf_matrix_file_test = os.path.join(
        "output_results/subset/plots", f"conf_matrix_test_{neurons}.png"
    )
    fig_conf_test.savefig(conf_matrix_file_test)
    plt.close(fig_conf_test)
    log_message(
        f"Confusion Matrix plot for test data saved to: {conf_matrix_file_test}"
    )

# Save the loss and accuracy plots
loss_accuracy_plot_file = "output_results/subset/plots/loss_accuracy_plot.png"
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(loss_accuracy_plot_file)
plt.close(fig)
log_message(f"Loss and accuracy plot saved to: {loss_accuracy_plot_file}")

log_message(
    f"\nBest accuracy achieved: {best_accuracy * 100:.2f}% with {best_neurons} neurons."
)
log_message(f"Total script runtime: {time.time() - start_time:.2f} seconds")
