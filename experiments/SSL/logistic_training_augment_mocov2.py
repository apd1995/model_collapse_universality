#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmented logistic regression training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
import torch.nn.init as init


# Initialize Weights and Biases
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="logistic-regression-SSL", entity="apd_stats", resume="allow", 
           name=f"{current_datetime}_mocov2_augment")

# Load the pre-saved features and labels
train_features = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/train/imagenet_resnet50_mocov2_2024-02-18_10-29-14.ckpt_train_features.pt')
train_labels = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/train/imagenet_resnet50_mocov2_2024-02-18_10-29-14.ckpt_train_labels.pt')
test_features = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/test/imagenet_resnet50_mocov2_2024-02-18_10-29-14.ckpt_test_features.pt')
test_labels = torch.load('/scratch/users/apd1995/SSL_checkpoints/checkpoints/test/imagenet_resnet50_mocov2_2024-02-18_10-29-14.ckpt_test_labels.pt')


# Move data to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_features, train_labels = train_features.to(device), train_labels.to(device)
test_features, test_labels = test_features.to(device), test_labels.to(device)

# Define the train and test datasets
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

# Define DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

input_dim = train_features.shape[1]  # 2048 dimensions
output_dim = len(torch.unique(train_labels))  # Number of classes (10 for CIFAR-10)

# Define a deterministic initialization function
def deterministic_init(m):
    if isinstance(m, nn.Linear):
        # Initialize weights with Xavier initialization (deterministically)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Initialize biases to zero
            init.constant_(m.bias, 0)

# Define training and evaluation functions
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

    return total_loss / len(train_loader), 100 * correct / total

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    return total_loss / len(test_loader), 100 * correct / total

# Training loop with 50 iterations
num_iterations = 10
epochs_per_iteration = 100

# Initialize model, loss function, and optimizer
model = LogisticRegressionModel(input_dim, output_dim).to(device)
model.apply(deterministic_init)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize with the original training labels and features
augmented_train_labels = train_labels.clone()
augmented_train_features = train_features.clone()

for iteration in range(num_iterations):
    logging.info(f"Starting Iteration {iteration + 1}/{num_iterations}")
    
    # Train for a specified number of epochs
    for epoch in range(epochs_per_iteration):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)
        # Log per-epoch training loss and accuracy to Weights and Biases (W&B)
        wandb.log({
            "Iteration": iteration + 1,
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy
        })
        logging.info(f"Iteration {iteration + 1}, Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%")
    
    # After training, evaluate the model on both training and test sets
    test_loss, test_accuracy = test_model(model, test_loader, criterion)

    # Log the metrics
    wandb.log({
        "Iteration": iteration + 1,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

    logging.info(f"Iteration {iteration + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}%, "
          f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%")
    
    # Set the random seed to the iteration number for reproducibility
    torch.manual_seed(iteration)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(iteration)

    # Predict on the training set and use predictions as new labels for the next iteration
    model.eval()
    new_train_labels = []
    new_train_features = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)
            
            # Sample from the multinomial distribution using the probabilities
            predicted_labels = torch.multinomial(probabilities, 1).squeeze(1)
            
            new_train_labels.append(predicted_labels)
            new_train_features.append(inputs)
            
    # Concatenate new labels and features
    new_train_labels = torch.cat(new_train_labels, dim=0)
    new_train_features = torch.cat(new_train_features, dim=0)
            
    # Move tensors to CPU before concatenating
    new_train_features_cpu = new_train_features.cpu()
    augmented_train_features_cpu = augmented_train_features.cpu()
    new_train_labels_cpu = new_train_labels.cpu()
    augmented_train_labels_cpu = augmented_train_labels.cpu()
    
    # Free the original GPU tensors to save memory
    del new_train_features, augmented_train_features
    del new_train_labels, augmented_train_labels
    torch.cuda.empty_cache()  # Free up memory after tensors have been moved to CPU
    
    # Concatenate on CPU to save GPU memory
    augmented_train_features = torch.cat((augmented_train_features_cpu, new_train_features_cpu), dim=0)
    augmented_train_labels = torch.cat((augmented_train_labels_cpu, new_train_labels_cpu), dim=0)
    
    # Move concatenated tensor back to GPU
    augmented_train_features = augmented_train_features.to(device)
    augmented_train_labels = augmented_train_labels.to(device)
    
    # Update the training dataset with the augmented features and labels
    train_dataset = TensorDataset(augmented_train_features, augmented_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


