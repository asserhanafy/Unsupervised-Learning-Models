import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path, random_state=None):
    # Load CSV
    data = pd.read_csv(file_path)

    # Fill missing values with column-wise mean
    data.fillna(data.mean(), inplace=True)

    # Split data: train 70%, val 15%, test 15%
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=random_state)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=random_state)

    # Convert to NumPy arrays
    X_train = train_data.values.astype(float)
    X_val = val_data.values.astype(float)
    X_test = test_data.values.astype(float)

    # Compute mean and std from training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # prevent division by zero

    # Standardize using training statistics
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test

# INITIALIZATION FUNCTIONS
def initialize_parameters(X, n_components, covariance_type='full', random_state=None):
    """
    Initialize GMM parameters using k-means++ style initialization.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Initialize means using k-means++ style
    means = np.zeros((n_components, n_features))
    means[0] = X[np.random.randint(n_samples)]
    
    for k in range(1, n_components):
        # Compute distances to nearest center
        distances = np.min([np.sum((X - means[j])**2, axis=1) 
                           for j in range(k)], axis=0)
        probs = distances / distances.sum()
        means[k] = X[np.random.choice(n_samples, p=probs)]
    
    # Initialize covariances
    covariances = initialize_covariances(X, n_components, covariance_type)
    
    # Initialize weights uniformly
    weights = np.ones(n_components) / n_components
    
    return means, covariances, weights


def initialize_covariances(X, n_components, covariance_type):
    """
    Initialize covariance matrices based on type.
    """
    n_features = X.shape[1]
    
    # Compute overall covariance of data
    data_cov = np.cov(X.T) + 1e-6 * np.eye(n_features)
    
    if covariance_type == 'full':
        # Each component has its own full covariance matrix
        covariances = np.array([data_cov for _ in range(n_components)])
        
    elif covariance_type == 'tied':
        # All components share the same covariance matrix
        covariances = data_cov
        
    elif covariance_type == 'diagonal':
        # Each component has diagonal covariance
        diag_cov = np.diag(np.diag(data_cov))
        covariances = np.array([diag_cov for _ in range(n_components)])
        
    elif covariance_type == 'spherical':
        # Each component has spherical covariance (scalar * I)
        variance = np.mean(np.diag(data_cov))
        covariances = np.array([variance for _ in range(n_components)])
    
    return covariances

# GAUSSIAN PROBABILITY DENSITY FUNCTION
def compute_gaussian_pdf(X, mean, covariance, covariance_type):
    """
    Compute Gaussian probability density function.
    """
    n_samples, n_features = X.shape
    diff = X - mean
    
    if covariance_type == 'full' or covariance_type == 'tied':
        # Full or tied covariance matrix
        cov_inv = np.linalg.inv(covariance)
        det = np.linalg.det(covariance)
        norm_const = 1.0 / np.sqrt((2 * np.pi)**n_features * det)
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        return norm_const * np.exp(exponent)
            
    elif covariance_type == 'diagonal':
        # Diagonal covariance
        cov_diag = np.diag(covariance)
        det = np.prod(cov_diag)
        norm_const = 1.0 / np.sqrt((2 * np.pi)**n_features * det)
        exponent = -0.5 * np.sum((diff**2) / cov_diag, axis=1)
        return norm_const * np.exp(exponent)
        
    elif covariance_type == 'spherical':
        # Spherical covariance (scalar variance)
        variance = covariance
        det = variance**n_features
        norm_const = 1.0 / np.sqrt((2 * np.pi)**n_features * det)
        exponent = -0.5 * np.sum(diff**2, axis=1) / variance
        return norm_const * np.exp(exponent)

# E-STEP: EXPECTATION
def e_step(X, means, covariances, weights, covariance_type):
    """
    E-step: Compute responsibilities (posterior probabilities).
    """
    n_samples = X.shape[0]
    n_components = len(means)
    
    # Compute weighted probabilities for each component
    weighted_prob = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        if covariance_type == 'tied':
            cov = covariances
        elif covariance_type in ['full', 'diagonal', 'spherical']:
            cov = covariances[k]
        # Compute probability density
        prob = compute_gaussian_pdf(X, means[k], cov, covariance_type)
        weighted_prob[:, k] = weights[k] * prob
    
    # Compute total probability (sum over components)
    total_prob = np.sum(weighted_prob, axis=1, keepdims=True)
    
    # Add small constant for numerical stability
    total_prob = np.maximum(total_prob, 1e-300)
    
    # Compute responsibilities (normalize)
    responsibilities = weighted_prob / total_prob
    
    # Compute log-likelihood
    log_likelihood = np.sum(np.log(total_prob.flatten()))
    
    return responsibilities, log_likelihood


# M-STEP: MAXIMIZATION
def m_step(X, responsibilities, covariance_type, reg_covar=1e-6):
    """
    M-step: Update parameters to maximize expected log-likelihood.
    reg_covar : float
        Regularization added to diagonal for numerical stability
    """
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    
    # Compute effective number of samples per component
    Nk = np.sum(responsibilities, axis=0)
    
    # Update weights
    weights = Nk / n_samples
    
    # Update means
    means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    
    # Update covariances based on type
    covariances = update_covariances(X, responsibilities, means, Nk, 
                                    covariance_type, reg_covar)
    
    return means, covariances, weights


def update_covariances(X, responsibilities, means, Nk, covariance_type, reg_covar):
    """
    Update covariance matrices based on type.
    Nk : array, shape (n_components,)
        Effective number of samples per component
    """
    n_samples, n_features = X.shape
    n_components = len(means)
    
    if covariance_type == 'full':
        # Full covariance for each component
        covariances = np.zeros((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            covariances[k] = np.dot(weighted_diff.T, diff) / Nk[k]
            # Add regularization for numerical stability
            covariances[k] += reg_covar * np.eye(n_features)
            
    elif covariance_type == 'tied':
        # Single shared covariance matrix
        covariances = np.zeros((n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            covariances += np.dot(weighted_diff.T, diff)
        covariances /= n_samples
        covariances += reg_covar * np.eye(n_features)
        
    elif covariance_type == 'diagonal':
        # Diagonal covariance for each component
        covariances = np.zeros((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            weighted_sq_diff = responsibilities[:, k:k+1] * (diff**2)
            variance = np.sum(weighted_sq_diff, axis=0) / Nk[k]
            covariances[k] = np.diag(variance + reg_covar)
            
    elif covariance_type == 'spherical':
        # Spherical covariance for each component (scalar variance)
        covariances = np.zeros(n_components)
        for k in range(n_components):
            diff = X - means[k]
            weighted_sq_diff = responsibilities[:, k:k+1] * (diff**2)
            covariances[k] = np.sum(weighted_sq_diff) / (Nk[k] * n_features)
            covariances[k] += reg_covar
    
    return covariances


# CONVERGENCE MONITORING
def check_convergence(log_likelihood_history, tol=1e-3, n_iter=10):
    """
    Check if EM algorithm has converged.
    """
    if len(log_likelihood_history) < n_iter:
        return False
    
    # Check relative change in log-likelihood
    current_ll = log_likelihood_history[-1]
    previous_ll = log_likelihood_history[-2]
    
    # Avoid division by zero
    if abs(previous_ll) < 1e-10:
        change = abs(current_ll - previous_ll)
    else:
        change = abs((current_ll - previous_ll) / previous_ll)
    
    return change < tol


# MAIN EM ALGORITHM
def fit_gmm(X, n_components, covariance_type='full', max_iter=100, 
            tol=1e-3, reg_covar=1e-6, random_state=None, verbose=False):
    # Initialize parameters
    means, covariances, weights = initialize_parameters(
        X, n_components, covariance_type, random_state
    )
    
    log_likelihood_history = []
    
    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        responsibilities, log_likelihood = e_step(
            X, means, covariances, weights, covariance_type
        )
        
        log_likelihood_history.append(log_likelihood)
        
        if verbose:
            print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.4f}")
        
        # Check convergence
        if iteration > 0 and check_convergence(log_likelihood_history, tol):
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            converged = True
            break
        
        # M-step: Update parameters
        means, covariances, weights = m_step(
            X, responsibilities, covariance_type, reg_covar
        )
        
        converged = False
    
    return {
        'means': means,
        'covariances': covariances,
        'weights': weights,
        'log_likelihood': log_likelihood_history[-1],
        'log_likelihood_history': log_likelihood_history,
        'converged': converged,
        'n_iter': len(log_likelihood_history)
    }


def predict_proba(X, means, covariances, weights, covariance_type):
    responsibilities, _ = e_step(X, means, covariances, weights, covariance_type)
    return responsibilities


def predict(X, means, covariances, weights, covariance_type):
    probabilities = predict_proba(X, means, covariances, weights, covariance_type)
    return np.argmax(probabilities, axis=1)