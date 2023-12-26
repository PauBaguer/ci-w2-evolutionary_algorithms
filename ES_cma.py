import numpy as np
import data
from deap import base, creator, tools, algorithms
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from cma import CMA

@tf.function
def predict(model, input):
    return model(input)

def obtain_mse(individual, X, y):
	weights = []
	start_idx = 0
	for layer in model.layers:
		num_params = np.prod(layer.get_weights()[0].shape)
		
		layer_weights = tf.reshape(individual[start_idx:start_idx + num_params], (layer.get_weights()[0].shape))
		layer.set_weights([layer_weights])

		weights.extend(layer.get_weights()[0].flatten())
		start_idx += num_params

	predictions = predict(model, X)
	mse = mean_squared_error(y, predictions)
	return mse
	

# Create a fitness function to minimize mean squared error
def fitness_function(individuals):
	global n_neurons
	mses = []
	for individual in individuals:
		mses.append(obtain_mse(individual, X_train, y_train))
	return tf.constant(mses)

max_epochs = 50
history = {
      'best_train': [],
	  'best_test': []
}
def logging_function(cma, logger):
	fitness = cma.best_fitness()
	history['best_train'].append(fitness)
	sol = cma.best_solution()
	history['best_test'].append(obtain_mse(sol, X_test, y_test))

	if cma.generation % 10 == 0:
		logger.info(f'Generation {cma.generation} - fitness {fitness}')
		print(f'Generation {cma.generation} - fitness {fitness}')

	if cma.termination_criterion_met or cma.generation == max_epochs:
		sol = cma.best_solution()
		logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')
		print(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')

np.random.seed(123)
X_train, X_test, y_train, y_test = data.generate_synthetic_data_reg(200, 0.01)

n_neurons = 10
n_weights = X_train.shape[1] * n_neurons + n_neurons

model = Sequential()
model.add(Dense(n_neurons, input_dim=X_train.shape[1], activation='relu', use_bias=False))
model.add(Dense(1, activation='linear', use_bias=False))

cma = CMA(
    initial_solution=np.random.uniform(-1, 1, n_weights),
    initial_step_size=1.0,
    fitness_function=fitness_function,
    callback_function=logging_function,
)

print('Starting CMA-ES')
with tf.device('/GPU:0'):
	best_solution, best_fitness = cma.search(max_epochs)
print('CMA-ES completed')

plt.plot(history['best_train'])
plt.plot(history['best_test'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['Training', 'Validation'])
plt.savefig('cma.png')