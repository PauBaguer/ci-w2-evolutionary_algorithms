import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

def generate_synthetic_data_reg(sample_size, noise_level, standard=True):
    # Generate synthetic regression dataset
    X, y = make_regression(
        n_samples=sample_size,
        n_features=2,  # Adjust the number of features as needed
        noise=noise_level,  # Standard deviation of Gaussian noise added to the output
        random_state=42
    )

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    if standard:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data', alpha=0.5)
    plt.title('Synthetic Regression Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Target (y)')  # Add colorbar to show the correspondence between color and target value
    plt.legend()
    plt.savefig('dataset.png')
    plt.clf()

    return X_train, X_val, X_test, y_train, y_val, y_test