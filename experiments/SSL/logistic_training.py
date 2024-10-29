#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:34:12 2024

@author: apratimdey
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# Initialize Weights and Biases
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="logistic-regression-SSL",
           entity="apd_stats",
           resume="allow",
           name=f"{current_datetime}")

# Load the pre-saved features and labels
train_features = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/train/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03.ckpt_train_features.pt')
train_labels = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/train/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03.ckpt_train_labels.pt')
test_features = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/test/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03.ckpt_test_features.pt')
test_labels = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/test/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03.ckpt_test_labels.pt')

# Move data to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the train and test datasets
train_dataset = TensorDataset(train_features.to(device), train_labels.to(device))
test_dataset = TensorDataset(test_features.to(device), test_labels.to(device))

# Define the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

input_dim = train_features.shape[1]  # 2048 dimensions
output_dim = len(torch.unique(train_labels))  # Number of classes (10 for CIFAR-10)

# Initialize model, move to GPU
model = LogisticRegressionModel(input_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

    # Log the loss and accuracy to Weights and Biases
    wandb.log({"train_loss": total_loss / len(train_loader),
               "train_accuracy": correct / total})

    return total_loss / len(train_loader), 100 * correct / total

# Function to evaluate the model
def test_model(model, test_loader, criterion):
    model.eval()  # Evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient calculation during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    # Log the loss and accuracy to Weights and Biases
    wandb.log({"test_loss": total_loss / len(test_loader),
               "test_accuracy": correct / total})

    return total_loss / len(test_loader), 100 * correct / total

# Training Loop
num_epochs = 10000
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
    test_loss, test_accuracy = test_model(model, test_loader, criterion)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

