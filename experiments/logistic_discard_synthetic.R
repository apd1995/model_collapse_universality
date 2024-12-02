library(ggplot2)

set.seed(123456) # Set a seed for reproducibility

# Parameters
n <- 10000  # number of samples
d <- 5     # number of features
iterations <- 50  # bootstrap iterations
experiments <- 10000  # number of experiments for averaging

# Function to compute logistic probabilities
logistic_prob <- function(X, coef, intercept) {
  linear <- X %*% coef + intercept
  1 / (1 + exp(-linear))
}

# Perform one experiment and collect coefficients
bootstrap_experiment <- function() {
  # Generate data
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  beta_true <- rnorm(d + 1) # True coefficients (including intercept)
  intercept_true <- beta_true[1]
  coef_true <- beta_true[-1]
  p <- logistic_prob(X, coef_true, intercept_true)
  y <- rbinom(n, 1, p)
  
  # Add intercept column to X
  X_with_intercept <- cbind(Intercept = 1, X)
  
  # Store squared differences for each iteration
  squared_differences <- matrix(0, nrow = iterations, ncol = d + 1)
  
  current_y <- y
  
  for (i in 1:iterations) {
    # Fit logistic regression using glm
    fit <- glm(current_y ~ ., data = data.frame(current_y, X), family = binomial)
    coef_fit <- coef(fit)  # Extract coefficients
    
    # Record squared differences with true coefficients
    squared_differences[i, ] <- (coef_fit - beta_true)^2
    
    # Generate new labels for bootstrap iteration
    prob <- logistic_prob(X, coef_fit[-1], coef_fit[1])
    current_y <- rbinom(n, 1, prob)
  }
  
  return(squared_differences)
}

# Perform multiple experiments and collect squared differences
all_squared_differences <- array(0, dim = c(iterations, d + 1, experiments))

for (exp in 1:experiments) {
  all_squared_differences[, , exp] <- bootstrap_experiment()
}

# Compute mean of squares of differences across experiments and dimensions
mean_squared_differences <- apply(all_squared_differences, 1, function(mat) mean(mat))

# Compute ratio of mean squared differences at iteration i to iteration 1
msd_ratios <- mean_squared_differences / mean_squared_differences[1]

# Create a data frame for ggplot
plot_data <- data.frame(
  Iteration = 1:iterations,
  Ratio = msd_ratios
)

# Plot using ggplot2
ggplot(plot_data, aes(x = Iteration, y = Ratio)) +
  geom_line(color = "blue") +
  geom_point(color = "blue") +
  labs(title = "Ratio of Mean Squared Error across Generations",
       x = "Iteration",
       y = "MSE Ratio") +
  theme_minimal()
