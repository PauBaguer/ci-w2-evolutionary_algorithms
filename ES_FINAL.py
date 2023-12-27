import numpy as np
import data
import sys
from deap import base, creator, tools, algorithms
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from cma import CMA
import numpy as np

tf.config.run_functions_eagerly(True)

@tf.function
def obtain_mse(individual, X, y):
	n_neurons = individual[0]
	wd = individual[1]
	weights = individual[2:]

	model = Sequential()
	model.add(Dense(n_neurons, input_dim=X.shape[1], activation='relu', use_bias=False))
	model.add(Dense(1, activation='linear', use_bias=False))

	start_idx = 0
	for layer in model.layers:
		num_params = np.prod(layer.get_weights()[0].shape)		
		layer_weights = tf.reshape(weights[start_idx:start_idx + num_params], (layer.get_weights()[0].shape))
		layer.set_weights([layer_weights])

		start_idx += num_params

	predictions = model(X)
	mse = mean_squared_error(y, predictions)
	return mse


@tf.function
def obtain_fitness(individual, X, y):
	n_neurons = individual[0]
	wd = individual[1]
	weights = individual[2:]

	model = Sequential()
	model.add(Dense(n_neurons, input_dim=X.shape[1], activation='relu', use_bias=False))
	model.add(Dense(1, activation='linear', use_bias=False))

	start_idx = 0
	for layer in model.layers:
		num_params = np.prod(layer.get_weights()[0].shape)		
		layer_weights = tf.reshape(weights[start_idx:start_idx + num_params], (layer.get_weights()[0].shape))
		layer.set_weights([layer_weights])

		start_idx += num_params

	predictions = model(X)
	mse = mean_squared_error(y, predictions) + wd * tf.reduce_sum(tf.square(weights))
	return mse
	

# Create a fitness function to minimize mean squared error
def fitness_function(individuals):
	global n_neurons
	fitness = []
	for individual in individuals:
		fitness.append(obtain_fitness(individual, X_train, y_train))
	fitness = tf.convert_to_tensor(fitness)
	return fitness

max_epochs = 500
history = {
	'best_fitness_train': [],
	'best_fitness_val': [],
	'best_mse_train': [],
	'best_mse_val': []
}
def logging_function(cma, logger):
	fitness = cma.best_fitness()
	history['best_fitness_train'].append(fitness)
	
	sol = cma.best_solution()
	history['best_fitness_val'].append(obtain_fitness(sol, X_val, y_val))

	history['best_mse_train'].append(obtain_mse(sol, X_train, y_train))
	history['best_mse_val'].append(obtain_mse(sol, X_val, y_val))

	if cma.generation % 10 == 0:
		print(f'Generation {cma.generation} - fitness {fitness}')

	if cma.termination_criterion_met or cma.generation == max_epochs:
		sol = cma.best_solution()
		print(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')

np.random.seed(123)
X_train, X_val, X_test, y_train, y_val, y_test = data.generate_synthetic_data_reg(2000, 0.05)

max_neurons = 100
n_weights = X_train.shape[1] * max_neurons + max_neurons
initial_solution = [50, 1e-4]
initial_solution.extend(np.random.uniform(-1, 1, n_weights))

bounds = [[1, max_neurons], [1e-10, np.inf]]
bounds.extend([[-np.inf, np.inf]] * n_weights)

cma = CMA(
	initial_solution=np.array(initial_solution),
	initial_step_size=1.0,
	enforce_bounds=np.array(bounds),
	fitness_function=fitness_function,
	callback_function=logging_function,
)

print('Starting CMA-ES')
with tf.device('/GPU:0'):
	best_solution, best_fitness = cma.search(max_epochs)
print('CMA-ES completed')

plt.plot(history['best_fitness_train'])
plt.plot(history['best_fitness_val'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['Training', 'Validation'])
plt.savefig('ES_training_fitness.png')
plt.clf()

plt.plot(history['best_mse_train'])
plt.plot(history['best_mse_val'])
plt.xlabel('Generation')
plt.ylabel('MSE')
plt.legend(['Training', 'Validation'])
plt.savefig('ES_training_mse.png')
plt.clf()

print('MSE on test set:', obtain_mse(best_solution, X_test, y_test))