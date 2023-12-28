
import numpy as np
import time
import data
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def do_MLP(k, n, h):
    np.random.seed(123)
    X_train, X_val, X_test, y_train, y_val, y_test = data.generate_synthetic_data_reg(10000, noise_level=n, hardness=h)
    n_neurons = 100
    epochs = 200
    learning_rate = 0.01


    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], activation='relu', use_bias=False))
    model.add(Dense(1, activation='linear', use_bias=False))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    end_time = time.time()
    train_time = end_time - start_time

    plt.figure(figsize=(4, 2))
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'MSE/MLP_training_mse_{k}_{n}_{h}.png')
    plt.clf()

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Test MSE: {mse}")

    return mse, train_time, n_neurons
