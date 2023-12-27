from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras, evolutionary_keras
from evolutionary_keras.optimizers import NGA
from MLP_old import MLP



# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    print()

    adam_optimizer = keras.optimizers.Adam()
    mlp = MLP(adam_optimizer)

    print("fit")
    mlp.fit(X_train_scaled, y_train, X_valid_scaled, y_valid, 20)

    GA_optimizer = NGA(population_size = 42, mutation_rate = 0.2)
    mlp = MLP(GA_optimizer)

    print("fit")
    mlp.fit(X_train_scaled, y_train, X_valid_scaled, y_valid, 20)