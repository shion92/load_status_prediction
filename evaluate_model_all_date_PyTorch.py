import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
import os
import json


# Define the MLP model structure
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


def plot_loss(train_loss, val_loss, neurons):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss for {neurons} Neurons")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = "output_results"
    plot_file = os.path.join(output_dir, f"loss_plot_{neurons}_neurons.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Loss plot saved to: {plot_file}")


def plot_accuracy(train_accuracy, val_accuracy, neurons):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy for {neurons} Neurons")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = "output_results"
    plot_file = os.path.join(output_dir, f"accuracy_plot_{neurons}_neurons.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Accuracy plot saved to: {plot_file}")


def load_and_evaluate_model(model_file, input_size, hidden_size, X_data, y_data):
    model = MLP(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred_labels = (y_pred >= 0.5).float()

    # Convert tensors to numpy arrays for classification report
    y_true_np = y_tensor.numpy()
    y_pred_np = y_pred_labels.numpy()

    # Calculate accuracy
    accuracy = (y_pred_labels == y_tensor).float().mean().item()

    # Generate classification report
    class_report = classification_report(
        y_true_np, y_pred_np, target_names=["Class 0", "Class 1"]
    )

    print(f"Model: {model_file}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", class_report)


def main():
    # Load and Preprocess the Data
    file_path = "data/train.csv"  # Update the path if necessary
    df = pd.read_csv(file_path)

    # Preprocessing and encoding
    df = df[
        (df["person_age"] <= 100) & (df["person_emp_length"] <= 60) &
        (df["person_income"] <= 1_000_000) & (df["loan_int_rate"] <= 35) &
        (df["cb_person_cred_hist_length"] <= 50)
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

    # Apply RandomUnderSampler to balance classes
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Check the class distribution after undersampling
    print("\nClass distribution after undersampling:")
    print(pd.Series(y_resampled).value_counts())

    # Standardize the Features
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # Split the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
    )

    # Define the hidden neuron options used in training
    neuron_options = [15]

    # Directory where models are saved
    output_dir = "output_results"
    neuron_options = [15]

    # Evaluate each model on the test data and plot losses
    for neurons in neuron_options:
        model_file = f"{output_dir}/model_{neurons}.pt"
        history_file = os.path.join(output_dir, f"training_history_{neurons}.json")

        # Load training and validation loss history
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
                train_loss = history.get("train_loss", [])
                val_loss = history.get("val_loss", [])
                train_accuracy = history.get("train_accuracy", [])
                val_accuracy = history.get("val_accuracy", [])

                # Plot the training and validation loss
                plot_loss(train_loss, val_loss, neurons)
                plot_accuracy(train_accuracy, val_accuracy, neurons)

        # Evaluate on test data
        print(f"\nEvaluating model with {neurons} neurons on test data...")
        load_and_evaluate_model(
            model_file,
            input_size=X_train.shape[1],
            hidden_size=neurons,
            X_data=X_test,
            y_data=y_test,
        )

        # Evaluate on full data
        print(f"\nEvaluating model with {neurons} neurons on all data...")
        load_and_evaluate_model(
            model_file,
            input_size=X_train.shape[1],
            hidden_size=neurons,
            X_data=X_resampled_scaled,
            y_data=y_resampled,
        )


if __name__ == "__main__":
    main()
