import numpy as np
import wandb
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Weights and Biases
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.init(project="logistic-regression-ML", entity="apd_stats", resume="allow", 
           name=f"{current_datetime}_wisconsin_augment")

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123456)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=10000, random_state=123456, penalty=None)

# Number of iterations for the loop
iterations = 50

# Initialize the augmented dataset with the original training data
X_train_augmented = X_train
y_train_augmented = y_train

# Iterative procedure
for i in range(iterations):
    logger.info(f"Iteration {i + 1}")
    
    # Fit the logistic regression model on the augmented training data
    model.fit(X_train_augmented, y_train_augmented)
    
    # Predict probabilities on the original training data (not augmented)
    y_train_probs = model.predict_proba(X_train)
    
    # Generate new labels by sampling from a Bernoulli distribution using predicted probabilities
    y_train_new = np.array([np.random.choice([0, 1], p=prob) for prob in y_train_probs])
    
    # Augment the training data with the newly predicted labels
    X_train_augmented = np.vstack([X_train_augmented, X_train])
    y_train_augmented = np.concatenate([y_train_augmented, y_train_new])
    
    # Compute training loss and accuracy on the augmented data
    train_loss = log_loss(y_train_augmented, model.predict_proba(X_train_augmented))
    train_accuracy = accuracy_score(y_train_augmented, model.predict(X_train_augmented))
    
    # Compute test loss and accuracy on the test data
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

# Finish logging
wandb.finish()

logger.info("Final model after 50 iterations trained with augmented labels.")
