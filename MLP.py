
import numpy as np
import time
import data
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def do_MLP(k, folder, sample_size, noise_level):
    np.random.seed(123)
    X_train, X_val, X_test, y_train, y_val, y_test = data.generate_synthetic_data_reg(sample_size, noise_level)
    n_neurons = 100
    epochs = 50
    learning_rate = 0.01


    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], activation='relu', use_bias=False))
    model.add(Dense(1, activation='linear', use_bias=False))
    optimizer = Adam(learning_rate=learning_rate, weight_decay=10e-4)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    end_time = time.time()
    train_time = end_time - start_time


    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/MLP_training_mse_{k}.png')

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Test MSE: {mse}")

    return mse, train_time, n_neurons
