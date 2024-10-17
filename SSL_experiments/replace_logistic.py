#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:13:24 2024

@author: apratimdey
"""

import numpy as np
import logging
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_name):
    """
    Load datasets based on the name provided.
    """
    if dataset_name == "titanic":
        from sklearn.datasets import fetch_openml
        data = fetch_openml('titanic', version=1, as_frame=True)
        X = data.data[['pclass', 'age', 'sex', 'fare']].copy()
        X['sex'] = X['sex'].map({'male': 0, 'female': 1})
        X = X.dropna()
        y = (data.target == '1').astype(int)
        y = y.loc[X.index]

    elif dataset_name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = data.data
        y = (data.target > 100).astype(int)  # Making a binary task: threshold diabetes levels
    
    elif dataset_name == "heart":
        from sklearn.datasets import fetch_openml
        data = fetch_openml('heart', version=1, as_frame='auto')
        X = data.data
        if hasattr(X, 'toarray'):  # Check if X is a sparse matrix
            X = X.toarray()  # Convert sparse matrix to dense array
        y = (data.target == 1).astype(int)
    
    elif dataset_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target
    
    else:
        raise ValueError("Unsupported dataset: " + dataset_name)
    
    return X, y

def replace_training_labels(X_train, y_train, model):
    """
    Replace the training labels with new labels generated from Bernoulli sampling
    based on the predicted probabilities from the model.
    """
    # Predict probabilities on the training data
    y_train_probs = model.predict_proba(X_train)
    
    # Generate new labels by sampling from a Bernoulli distribution
    y_train_new = np.array([np.random.choice([0, 1], p=prob) for prob in y_train_probs])
    
    return y_train_new

def main(dataset_name, iterations=50):
    # Initialize wandb for experiment tracking
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="logistic-regression-ML",entity="apd_stats", resume="allow",
               name=f"{current_datetime}_{dataset_name}_replace")
    
    # Load the dataset
    logger.info(f"Loading dataset: {dataset_name}")
    X, y = load_dataset(dataset_name)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123456)

    # Initialize logistic regression model
    model = LogisticRegression(max_iter=10000, random_state=123456, penalty = None)

    # Iterative procedure
    for i in range(iterations):
        logger.info(f"Iteration {i + 1}")
        
        # Fit the logistic regression model on the augmented training data
        model.fit(X_train, y_train)
        
        # Replace training labels using the Bernoulli-sampled labels
        y_train = replace_training_labels(X_train, y_train, model)
        
        # Calculate training and test metrics
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Log metrics to WandB
        wandb.log({
            "iteration": i + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
        logger.info(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Finish logging
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset replacement with Logistic Regression")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (titanic, diabetes, heart, breast_cancer, adult, wisconsin_breast_cancer)")
    parser.add_argument('--iterations', type=int, default=50, help="Number of iterations for augmentation")
    args = parser.parse_args()
    
    main(args.dataset, args.iterations)