import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import json
import os
import numpy as np
import sys

# Redirect output to a file in append mode
output_file = open("output_log.txt", "a")
sys.stdout = output_file


# Define the MLP Model with PyTorch
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


def calculate_accuracy(y_pred, y_true):
    # Calculate accuracy: compare predicted labels to true labels
    y_pred_labels = (y_pred >= 0.5).float()  # Convert probabilities to 0/1
    correct = (y_pred_labels == y_true).float().sum()  # Count correct predictions
    accuracy = correct / y_true.shape[0]
    return accuracy


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
    X = df.drop(["loan_status", "id"], axis=1)
    y = df["loan_status"]

    # Apply SMOTE to balance the classes in the target variable
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Check the class distribution after SMOTE
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())

    # Standardize the Features
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)

    # Split the Resampled Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create PyTorch DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,  # Set num_workers to 0 for compatibility
    )

    # Directory for saving outputs
    output_dir = "output_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define early stopping parameters
    patience = 50  # Number of epochs to wait for improvement
    min_delta = 0.0001  # Minimum change to qualify as improvement

    # Train and Evaluate the Model with Different Hidden Neurons
    best_accuracy = 0
    best_neurons = 0
    neuron_options = [15]

    for neurons in neuron_options:
        # Start timing for the current model
        start_model_time = time.time()

        print(f"\nTraining model with {neurons} neurons in the hidden layer...")
        sys.stdout.flush()  # Flush to ensure immediate write to file

        model = MLP(input_size=X_train.shape[1], hidden_size=neurons)
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Initialize variables for early stopping
        best_val_loss = np.inf
        patience_counter = 0
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        # Training loop with early stopping
        num_epochs = 1000
        for epoch in range(num_epochs):
            model.train()
            epoch_train_acc = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Reset gradients
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                # Calculate batch accuracy
                batch_train_acc = calculate_accuracy(y_pred, y_batch).item()
                epoch_train_acc += batch_train_acc

            # Average training accuracy for the epoch
            epoch_train_acc /= len(train_loader)
            train_acc_history.append(epoch_train_acc)

            # Calculate validation loss and accuracy
            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test_tensor)
                val_loss = criterion(y_test_pred, y_test_tensor).item()
                val_loss_history.append(val_loss)

                # Calculate validation accuracy
                val_acc = calculate_accuracy(y_test_pred, y_test_tensor).item()
                val_acc_history.append(val_acc)

                # Check for early stopping condition
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model state
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, f"model_{neurons}.pt"),
                    )
                else:
                    patience_counter += 1

            # Store training loss
            train_loss_history.append(loss.item())

            # Print progress
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                print(
                    f"Epoch [{epoch}/{num_epochs}], "
                    f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {epoch_train_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%"
                )
                sys.stdout.flush()

            # Break if early stopping condition is met
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} for {neurons} neurons.")
                sys.stdout.flush()
                break

        # Evaluate the model on the test set
        y_test_pred = (y_test_pred >= 0.5).float()  # Threshold at 0.5
        accuracy = (y_test_pred == y_test_tensor).float().mean().item()

        print(f"Accuracy with {neurons} neurons: {accuracy * 100:.2f}%")
        print(
            f"Time taken for {neurons} neurons: {time.time() - start_model_time:.2f} seconds"
        )
        sys.stdout.flush()

        # Save training and validation history
        history_file = os.path.join(output_dir, f"training_history_{neurons}.json")
        with open(history_file, "w") as f:
            json.dump(
                {
                    "train_loss": train_loss_history,
                    "val_loss": val_loss_history,
                    "train_accuracy": train_acc_history,
                    "val_accuracy": val_acc_history,
                },
                f,
                indent=4,
            )

        # Update best accuracy and neurons
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_neurons = neurons

    # Display total runtime
    print(
        f"\nBest accuracy achieved: {best_accuracy * 100:.2f}% with {best_neurons} neurons."
    )
    print(f"Total script runtime: {time.time() - start_time:.2f} seconds")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

    # Close the output file at the end
    output_file.close()
