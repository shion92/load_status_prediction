import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Adding seaborn for better visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import os


# Define the MLP model structure to match the saved models
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


def load_and_evaluate_model(model_file, input_size, hidden_size, X_data, y_data):
    """
    Load the model from a .pt file and evaluate it on the provided data.
    """
    # Load model structure
    model = MLP(input_size=input_size, hidden_size=hidden_size)

    # Load model weights
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set the model to evaluation mode

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        y_pred = model(X_tensor)
        y_pred = (y_pred >= 0.5).float().numpy().flatten()  # Apply threshold

    # Calculate accuracy
    accuracy = accuracy_score(y_data, y_pred) * 100

    # Generate classification report
    class_report = classification_report(
        y_data, y_pred, target_names=["Class 0", "Class 1"]
    )

    print(f"Model: {model_file}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification Report:\n", class_report)


def main():
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

    # Define the hidden neuron options used in training
    neuron_options = [5, 10, 15, 20]

    # Directory where models are saved
    output_dir = "output_results"

    # Evaluate each model on the test data
    for neurons in neuron_options:
        model_file = f"{output_dir}/model_{neurons}_neurons.pt"

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
            X_data=X_scaled,
            y_data=y,
        )


if __name__ == "__main__":
    main()
