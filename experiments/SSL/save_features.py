#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:34:07 2024

@author: apratimdey
"""

import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Load the checkpoint that contains the SSL backbone
directory = '/scratch/users/apd1995/SSL_checkpoints/checkpoints/'
files = os.listdir(directory)
files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
for filename in files:
    checkpoint = torch.load(directory + filename)
    
    # Extract the backbone state_dict (strip "backbone." prefix)
    backbone_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("backbone.")}
    
    # Check if CUDA is available and move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet50 architecture (without pretrained weights, as we're loading from the SSL checkpoint)
    resnet50 = models.resnet50(weights=None).to(device)
    
    # Replace the fully connected (classification) layer with an identity layer for feature extraction
    resnet50.fc = torch.nn.Identity()
    
    # Load the backbone weights into ResNet50
    resnet50.load_state_dict(backbone_state_dict, strict=False)
    
    # Set the model to evaluation mode
    resnet50.eval()
    
    # Define transformations for ImageNet (resize, center crop, and normalize)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download and load the CIFAR-10 training dataset
    cifar10_train = datasets.CIFAR10(root='/scratch/users/apd1995/SSL_checkpoints/data', train=True, download=True, transform=transform)
    
    # Download and load the CIFAR-10 test dataset
    cifar10_test = datasets.CIFAR10(root='/scratch/users/apd1995/SSL_checkpoints/data', train=False, download=True, transform=transform)
    
    # Create DataLoader for batch processing
    train_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=32, shuffle=False)
    
    # Function to extract features using ResNet50
    def extract_features(dataloader, model):
        features_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                # Move images and labels to the GPU
                images = images.to(device)
                labels = labels.to(device)
                # Extract features using the ResNet50 backbone on the GPU
                features = model(images)
                features_list.append(features.cpu())  # Move back to CPU to save memory
                labels_list.append(labels.cpu())      # Move back to CPU to save memory    
        # Concatenate all features and labels into tensors
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
    
    # Extract features from the CIFAR-10 training dataset
    train_features, train_labels = extract_features(train_loader, resnet50)
    
    # Extract features from the CIFAR-10 test dataset
    test_features, test_labels = extract_features(test_loader, resnet50)
    
    # Save the features
    train_directory = directory + 'train/'
    test_directory = directory + 'test/'
    torch.save(train_features, train_directory+filename+'_train_features.pt')
    torch.save(train_labels, train_directory+filename+'_train_labels.pt')
    torch.save(test_features, test_directory+filename+'_test_features.pt')
    torch.save(test_labels, test_directory+filename+'_test_labels.pt')

# Convert extracted features to NumPy arrays for further processing
# train_features_np = train_features.numpy()
# train_labels_np = train_labels.numpy()

# test_features_np = test_features.numpy()
# test_labels_np = test_labels.numpy()

# # Initialize and train a logistic regression model
# clf = LogisticRegression(penalty=None, max_iter=5000)
# clf.fit(train_features_np, train_labels_np)

# # Make predictions on the test set
# train_labels_pred = clf.predict(train_features_np)
# test_labels_pred = clf.predict(test_features_np)

# # Calculate accuracy of the logistic regression model
# accuracy = accuracy_score(test_labels_np, test_labels_pred)
# print(f"Logistic Regression Accuracy: {accuracy:.4f}")
