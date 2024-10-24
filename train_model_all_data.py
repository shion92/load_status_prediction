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

# Start measuring the total runtime
start_time = time.time()

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

# Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
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
neuron_options = [2, 4, 6, 8, 10, 15, 20]

fig, axes = plt.subplots(len(neuron_options), 2, figsize=(14, 4 * len(neuron_options)))
fig.suptitle("Training Performance for Different Neurons", fontsize=16)

for i, neurons in enumerate(neuron_options):
    print(f"Training model with {neurons} neurons in the hidden layer...")
    # Start timing for the current model
    start_model_time = time.time()
    model = build_model(neurons)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=32,
        verbose=0,
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy with {neurons} neurons: {accuracy * 100:.2f}%")
    print(
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

    print(
        f"\nClassification report for {neurons} neurons:\n",
        classification_report(y_test, predicted),
    )

    # Display Confusion Matrix
    conf_mat = confusion_matrix(y_test, predicted)
    print(f"Confusion Matrix for {neurons} neurons:\n", conf_mat)

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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(
    f"\nBest accuracy achieved: {best_accuracy * 100:.2f}% with {best_neurons} neurons."
)
print(f"Total script runtime: {time.time() - start_time:.2f} seconds")
