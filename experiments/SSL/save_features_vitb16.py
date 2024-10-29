#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:34:07 2024

@author: apratimdey
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# Load the checkpoint that contains the SSL backbone
directory = '/scratch/users/apd1995/SSL_checkpoints/checkpoints/'
filename = 'imagenet_vitb16_mae_2024-02-25_19-57-30.ckpt'

checkpoint = torch.load(directory + filename)
state_dict = checkpoint['state_dict']

# Remove 'backbone.' and 'vit.' prefixes and filter out decoder and classifier head keys
filtered_state_dict = {k.replace('backbone.vit.', '').replace('backbone.', ''): v 
                       for k, v in state_dict.items() 
                       if not k.startswith('decoder') and 'mask_token' not in k and 'classification_head' not in k}

# Check if CUDA is available and move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ViT-B/16 model
vitb16 = timm.create_model('vit_base_patch16_224', pretrained=False).to(device)

# Remove the classification head (fc layer) for feature extraction
vitb16.head = torch.nn.Identity()

# Load the filtered state dict into the model
vitb16.load_state_dict(filtered_state_dict, strict=False)


# Set the model to evaluation mode
vitb16.eval()

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
train_features, train_labels = extract_features(train_loader, vitb16)

# Extract features from the CIFAR-10 test dataset
test_features, test_labels = extract_features(test_loader, vitb16)

# Save the features
train_directory = directory + 'train/'
test_directory = directory + 'test/'
torch.save(train_features, train_directory+filename+'_train_features.pt')
torch.save(train_labels, train_directory+filename+'_train_labels.pt')
torch.save(test_features, test_directory+filename+'_test_features.pt')
torch.save(test_labels, test_directory+filename+'_test_labels.pt')