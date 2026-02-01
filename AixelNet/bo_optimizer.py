from scipy.stats import uniform, randint
import json
import os
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

# Define the search space for hyperparameters
search_space = {
    'lr': uniform(1e-5, 1e-2),         # Learning rate between 1e-5 and 1e-2
    'batch_size': [32, 64, 128, 256],    # Batch size from the set {32, 64, 128, 256}
    'num_epoch': randint(10, 20),      # Number of epochs between 10 and 20
}

# Define the surrogate models for performance and cost
class SurrogateModel:
    def __init__(self):
        # Kernel: constant * RBF kernel for Gaussian Process
        self.kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
    
    def fit(self, X, y):
        """ Fit the model to data (X, y) """
        self.gpr.fit(X, y)
    
    def predict(self, X):
        """ Predict with the fitted model """
        return self.gpr.predict(X, return_std=True)

# Define the acquisition function
def acquisition(mu, sigma, mu_cost):
    """
    Cost-sensitive acquisition function.
    This function balances expected improvement and cost for new hyperparameter selection.
    """
    expected_improvement = np.maximum(0, mu - np.min(mu))  # Improvement over the best observed performance
    acquisition_values = expected_improvement / (mu_cost + 1e-5)  # Balance performance with cost
    return acquisition_values

# Define the Bayesian Optimization (BO) process
def optimize_hyperparameters(meta_features, param_space=search_space, history_path="history_data.json"):
    """
    Optimizes hyperparameters using Bayesian Optimization (BO).
    Trains surrogate models and guides the search towards promising configurations.
    
    meta_features: Features of the current table to guide the BO process.
    param_space: Hyperparameter search space.
    history_path: Path to the file storing historical BO data.
    
    Returns the optimal configuration.
    """
    # Load historical optimization results from file
    history_data = load_history_data(history_path)
    
    if not history_data:
        print("No historical data found. Using default hyperparameters.")
        return {
            'lr': 1e-4,
            'batch_size': 64,
            'num_epoch': 200
        }

    # Extract historical data into arrays
    meta_arr = np.array([h['meta_features'] for h in history_data])
    config_arr = np.array([h['config'] for h in history_data])
    loss_arr = np.array([h['loss'] for h in history_data])
    cost_arr = np.array([h['cost'] for h in history_data])

    # Train performance and cost surrogate models
    perf_surrogate = SurrogateModel()
    cost_surrogate = SurrogateModel()
    X_input = np.concatenate([meta_arr, config_arr], axis=1)  # Concatenate meta features and configurations

    perf_surrogate.fit(X_input, loss_arr)  # Train performance model
    cost_surrogate.fit(X_input, cost_arr)  # Train cost model
    
    # Generate candidate hyperparameter configurations
    param_combinations = list(itertools.product(*[param_space[key] for key in param_space]))  # Sample combinations of hyperparameters
    test_configs = np.array([list(config) for config in param_combinations])

    # Generate the feature space for the candidate configurations
    test_inputs = np.concatenate([np.tile(meta_features, (len(test_configs), 1)), test_configs], axis=1)

    # Predict performance and cost using the surrogate models
    mu_loss, sigma_loss = perf_surrogate.predict(test_inputs)
    mu_cost, _ = cost_surrogate.predict(test_inputs)

    # Compute the acquisition values using the expected improvement formula
    acq_values = acquisition(mu_loss, sigma_loss, mu_cost)
    best_idx = np.argmax(acq_values)  # Select the best configuration based on the acquisition values
    best_config = param_combinations[best_idx]

    # Store the new configuration in the history data
    new_entry = {
        'meta_features': meta_features.tolist(),
        'config': best_config,
        'loss': mu_loss[best_idx],
        'cost': mu_cost[best_idx]
    }
    history_data.append(new_entry)
    save_history_data(history_data, history_path)  # Save updated history data
    
    return best_config

# Helper functions for loading and saving history data
def load_history_data(history_path):
    """ Load historical data from a JSON file. """
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    else:
        return []

def save_history_data(history_data, history_path):
    """ Save updated history data to a JSON file. """
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=4)
