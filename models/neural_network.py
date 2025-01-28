import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from utils.preprocessing_pipeline import preprocessor



LAYER_1 = 128
LAYER_2 = 64

class CompasNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CompasNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_dim, LAYER_1)
        self.bn1 = nn.BatchNorm1d(LAYER_1)
        self.hidden2 = nn.Linear(LAYER_1, LAYER_2)
        self.bn2 = nn.BatchNorm1d(LAYER_2)
        self.output = nn.Linear(LAYER_2, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = torch.relu(self.bn1(self.hidden1(x)))
        x = torch.relu(self.bn2(self.hidden2(x)))
        x = self.output(x)
        x = self.softmax(x)
        return x


class CompasNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=20, batch_size=32, learning_rate=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = CompasNeuralNetwork(input_dim)
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()

    def fit(self, X, y):

        y = y.to_numpy() if hasattr(y, "to_numpy") else y


        self.classes_ = torch.unique(torch.tensor(y, dtype=torch.long)).numpy()

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)  # CrossEntropyLoss expects class labels

        # DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # for mse loss
                y_one_hot = torch.nn.functional.one_hot(batch_y, num_classes=2).float()  # Shape: (batch_size, 2)

                # loss = self.criterion(outputs, batch_y)
                #use for mse
                loss = self.criterion(outputs, y_one_hot)
                
                loss.backward()
                optimizer.step()
                print (f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        return self

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        return predictions.numpy()

    def predict_proba(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()




# Load best parameters
parameters_path = 'results/nn_best_params.txt'
best_params = {}

if os.path.exists(parameters_path):
    with open(parameters_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Best Parameters:"):
                best_params = eval(line.split(":", 1)[1].strip())

# Define default parameters
default_params = {
    'input_dim': 12,  # You can dynamically determine this later
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001
}

# Merge best parameters
best_params = {**default_params, **{k.split("__")[1]: v for k, v in best_params.items()}}

# Load training data to infer input dimensions
train_data = pd.read_csv('data/train/train_compas-scores-two-years.csv')
X_train = train_data.drop(columns=['two_year_recid'])  # Replace with your target column
input_dim = preprocessor.fit_transform(X_train).shape[1]
best_params['input_dim'] = input_dim

# Create PyTorch classifier with best parameters
pytorch_classifier = CompasNeuralNetworkClassifier(**best_params)

# Define pipeline
nn_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),  # Reuse preprocessing pipeline
    ('classifier', pytorch_classifier)
])
