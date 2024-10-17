import numpy as np
import wandb
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Weights and Biases
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="logistic-regression-ML", entity="apd_stats", resume="allow", 
           name=f"{current_datetime}_wisconsin_replace")
# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123456)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=10000, random_state=123456, penalty = None)

# Number of iterations for the loop
iterations = 50

# Iterative procedure
for i in range(iterations):
    logger.info(f"Iteration {i + 1}")
    
    # Fit the logistic regression model on the training data
    model.fit(X_train, y_train)
    
    # Predict probabilities on the training data
    y_train_probs = model.predict_proba(X_train)
    
    # Generate new labels by sampling from a Bernoulli distribution using predicted probabilities
    y_train_new = np.array([np.random.choice([0, 1], p=prob) for prob in y_train_probs])
    
    # Compute training loss and accuracy
    train_loss = log_loss(y_train, y_train_probs)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    # Compute test loss and accuracy
    y_test_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_test_probs)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Log metrics to WandB
    wandb.log({
        "iteration": i + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    logger.info(f"Training loss: {train_loss}, Training accuracy: {train_accuracy}")
    logger.info(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
    
    # Replace the training labels with the new sampled labels
    y_train = y_train_new

# Finish logging
wandb.finish()

logger.info("Final model after 50 iterations trained.")
