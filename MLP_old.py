import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, optimizer):
        self.optimizer = optimizer


    def fit(self, X_train, y_train, X_val, y_val, epochs):

        # # Split data
        # X_train, X_val = X[train_index], X[val_index]
        # y_train, y_val = y[train_index], y[val_index]
        # # Create the model
        # input_dim = X.shape[1]
        # model = Sequential()
        # model.add(Dense(units=64, activation='relu', input_dim=input_dim))  # Hidden layer with ReLU activation
        # model.add(Dense(units=32, activation='relu'))  # Hidden layer with ReLU activation
        # model.add(Dense(units=1, activation='linear'))  # Output layer for regression
        # model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model

        model = Sequential()
        model.add(Dense(64, kernel_initializer='uniform', input_shape=(8,))),
        model.add(Flatten())
        model.add(Activation('softmax'))
        """
        # Instantiate an optimizer.
        optimizer = tf.keras.optimizers.Adam()

        # Iterate over the batches of a dataset.
        for x, y in dataset:
            # Open a GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = model(x)
                # Loss value for this batch.
                loss_value = loss_fn(y, logits)

            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss_value, model.trainable_weights)

            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        """

        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)




        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=1)
        history_dict = history.history

        # # Predict
        # y_pred = model.predict(X_val, verbose=0)
        # p = pearsonr(y_val, y_pred.reshape(1, -1)[0])[0]
        # print("Number of fold: ", fold_idx + 1)
        # print("Pearson correlation: ", p)

        # Plot the training and validation accuracy curves
        plt.figure(figsize=(4, 2))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

