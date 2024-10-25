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

# Start measuring the total runtime
start_time = time.time()

# Create directories for saving outputs
os.makedirs("output_results/plots", exist_ok=True)
os.makedirs("output_results/models", exist_ok=True)
output_log = "output_results/non_torch_output_log.txt"


# Function to write to output log
def log_message(message):
    with open(output_log, "a") as f:
        f.write(message + "\n")
    print(message)


# Load and Preprocess the Data
file_path = "data/train.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# Remove Outliers
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
X = df.drop("loan_status", axis=1)
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
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)


# Define the MLP Model with One Hidden Layer
def build_model(hidden_neurons):
    model = Sequential(
        [
            Dense(hidden_neurons, input_dim=X_train.shape[1], activation="sigmoid"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )
    return model


# Train and Evaluate the Model with Different Hidden Neurons
best_accuracy = 0
best_neurons = 0
neuron_options = [5, 10, 15, 20]

fig, axes = plt.subplots(len(neuron_options), 2, figsize=(14, 4 * len(neuron_options)))
fig.suptitle("Training Performance for Different Neurons", fontsize=16)

for i, neurons in enumerate(neuron_options):
    log_message(f"\nTraining model with {neurons} neurons in the hidden layer...")

    # Start timing for the current model
    start_model_time = time.time()
    model = build_model(neurons)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=128,
        verbose=0,
    )

    # Save the trained model
    model_file = os.path.join("output_results/models", f"model_{neurons}.h5")
    model.save(model_file)
    log_message(f"Model saved to: {model_file}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log_message(f"Accuracy with {neurons} neurons: {accuracy * 100:.2f}%")
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

    # Make Predictions and Classification Report
    predicted_y = model.predict(X_test)
    predicted = squeeze(predicted_y)
    predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])

    classification_rep = classification_report(y_test, predicted)
    log_message(f"\nClassification report for {neurons} neurons:\n{classification_rep}")

    # Display Confusion Matrix
    conf_mat = confusion_matrix(y_test, predicted)
    log_message(f"Confusion Matrix for {neurons} neurons:\n{conf_mat}")

    # Confusion Matrix Plot
    fig_conf, ax_conf = plt.subplots(1, 2, figsize=(10, 4))
    fig_conf.suptitle(f"Confusion Matrix for {neurons} Neurons", fontsize=14)

    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, ax=ax_conf[0], cmap="Blues"
    )
    ax_conf[0].set_title("Confusion Matrix (Raw)")

    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, normalize="true", ax=ax_conf[1], cmap="Blues"
    )
    ax_conf[1].set_title("Confusion Matrix (Normalized)")

    # Save Confusion Matrix Plot
    conf_matrix_file = os.path.join(
        "output_results/plots", f"conf_matrix_{neurons}.png"
    )
    fig_conf.savefig(conf_matrix_file)
    plt.close(fig_conf)
    log_message(f"Confusion Matrix plot saved to: {conf_matrix_file}")

# Save the loss and accuracy plots
loss_accuracy_plot_file = "output_results/plots/loss_accuracy_plot.png"
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(loss_accuracy_plot_file)
plt.close(fig)
log_message(f"Loss and accuracy plot saved to: {loss_accuracy_plot_file}")

log_message(
    f"\nBest accuracy achieved: {best_accuracy * 100:.2f}% with {best_neurons} neurons."
)
log_message(f"Total script runtime: {time.time() - start_time:.2f} seconds")
