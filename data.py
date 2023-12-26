import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

def generate_synthetic_data_clf(sample_size, noise_level, problem_hardness, standard=True):
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=sample_size,
        n_features=10,  # Adjust the number of features as needed
        n_informative=5,  # Number of informative features
        n_redundant=2,  # Number of redundant features
        n_clusters_per_class=1,  # Number of clusters per class
        random_state=42,
        flip_y=noise_level,  # Amount of label noise
        class_sep=problem_hardness  # Larger values make the problem harder
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if standard:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def generate_synthetic_data_reg(sample_size, noise_level, standard=True):
    # Generate synthetic regression dataset
    X, y = make_regression(
        n_samples=sample_size,
        n_features=2,  # Adjust the number of features as needed
        noise=noise_level,  # Standard deviation of Gaussian noise added to the output
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if standard:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
    

# Example usage
# # Set parameters
# sample_size = 20000  # Vary this for the training set
# noise_level = 0.1  # Adjust as needed
# problem_hardness = 2.0  # Adjust as needed

# # Generate synthetic classification data
# X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_data_clf(sample_size, noise_level, problem_hardness)
