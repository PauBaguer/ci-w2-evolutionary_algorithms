
import numpy as np
import data
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(123)
X_train, X_test, y_train, y_test = data.generate_synthetic_data_reg(200, 0.01)
n_neurons = 10
epochs = 100
learning_rate = 0.001

#n_neurons_arr = [5, 10, 20, 30, 40, 50, 75, 100, 500, 1000]
n_neurons_arr = np.array([5, 10, 20, 30, 40, 50, 75, 100, 500, 1000]) * 1000
mse_arr = []
for n_neurons in n_neurons_arr:
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X_train.shape[1], activation='relu', use_bias=False))
    model.add(Dense(1, activation='linear', use_bias=False))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1) #validation_data=(X_val, y_val)


    plt.figure(figsize=(4, 2))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mse_arr.append(mse)

    print(f"Test MSE: {mse}")


plt.scatter(n_neurons_arr, mse_arr)
plt.title(f'Test MSE')
plt.xlabel('Nº of neurons')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
