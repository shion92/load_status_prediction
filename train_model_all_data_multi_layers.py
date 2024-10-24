import pandas as pd
import matplotlib.pyplot as plt
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
import os
import json


# Define the DNN Model with PyTorch
class DNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super(DNN, self).__init__()
        layers = []

        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main():
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

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create PyTorch DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Directory for saving outputs
    output_dir = "output_results/layers"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define model configurations
    num_layers_options = [2, 3, 4, 5, 7, 10]  # Different number of layers
    hidden_size = 10  # 10 neurons per hidden layer

    for num_layers in num_layers_options:
        # Start timing for the current model
        start_model_time = time.time()

        print(
            f"\nTraining DNN with {num_layers} layers and {hidden_size} neurons per layer..."
        )
        model = DNN(
            input_size=X_train.shape[1], num_layers=num_layers, hidden_size=hidden_size
        )
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model
        num_epochs = 50
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Reset gradients
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

        # Evaluate the model on the test set
        with torch.no_grad():
            y_test_pred = model(X_test_tensor)
            y_test_pred = (y_test_pred >= 0.5).float()  # Threshold at 0.5
            accuracy = (y_test_pred == y_test_tensor).float().mean().item()

        print(f"Accuracy with {num_layers} layers: {accuracy * 100:.2f}%")
        print(
            f"Time taken for {num_layers} layers: {time.time() - start_model_time:.2f} seconds"
        )

        # Save the current model
        model_file = os.path.join(output_dir, f"model__{num_layers}.pt")
        torch.save(model.state_dict(), model_file)

        # Classification report and confusion matrix
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        report_file = os.path.join(
            output_dir, f"classification_report_{num_layers}_layers.json"
        )
        with open(report_file, "w") as f:
            json.dump(class_report, f, indent=4)

        conf_mat = confusion_matrix(y_test, y_test_pred)
        print(f"Confusion Matrix for {num_layers} layers:\n", conf_mat)

        # Save confusion matrix as an image
        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, cmap="Blues")
        plt.title(f"Confusion Matrix for {num_layers} Layers")
        plt.savefig(
            os.path.join(output_dir, f"confusion_matrix_{num_layers}_layers.png")
        )
        plt.close()

    # Display total runtime
    print(f"\nTotal script runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
