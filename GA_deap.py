import numpy as np
import data
from deap import base, creator, tools, algorithms
import tensorflow as tf
from deap import tools
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# Create a fitness function to minimize mean squared error
def fitness_function(individual, X, y):
    global n_neurons
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=X.shape[1], activation='relu', use_bias=False))
    model.add(Dense(1, activation='linear', use_bias=False))
    
    # Set the weights of the model
    weights = []
    start_idx = 0
    for layer in model.layers:
        num_params = np.prod(layer.get_weights()[0].shape)
        layer_weights = individual[start_idx:start_idx + num_params].reshape(layer.get_weights()[0].shape)
        layer.set_weights([layer_weights])

        weights.extend(layer.get_weights()[0].flatten())
        start_idx += num_params

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return mse,


np.random.seed(123)
X_train, X_test, y_train, y_test = data.generate_synthetic_data_reg(200, 0.01)



n_neurons = 10
n_weights = X_train.shape[1] * n_neurons + n_neurons


history = tools.History()
# Define the problem as a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

# Set up the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

# Define the fitness function with data as an argument
toolbox.register("evaluate", fitness_function, X=X_train, y=y_train)

# Create an initial population
population = toolbox.population(n=10)
history.update(population)

# Define the number of generations
n_gen = 50
# algorithms.eaMuPlusLambda(population, toolbox, mu=5, lambda_=10, cxpb=0.7, mutpb=0.2, ngen=n_gen, stats=None, halloffame=None)

# Store the mean and best fitness values
mean_fitness_values = []
best_fitness_values = []

# Run the genetic algorithm
for gen in range(n_gen):
    algorithms.eaMuPlusLambda(population, toolbox, mu=5, lambda_=10, cxpb=0.7, mutpb=0.2, ngen=1, stats=None, halloffame=None)
    
    print(f"----------------- Generation {gen+1} -----------------")
    best = tools.selBest(population, k=1)[0]
    print(f"Best fitness: {best.fitness.values[0]}")
    
    fitness_values = [ind.fitness.values[0] for ind in population]
    
    # Compute the mean fitness value
    mean_fitness = sum(fitness_values) / len(fitness_values)
    print(f"Mean fitness: {mean_fitness}")
    
    # Store the mean and best fitness values
    mean_fitness_values.append(mean_fitness)
    best_fitness_values.append(best.fitness.values[0])

# Print the best individual and its fitness value
best_individual = tools.selBest(population, k=1)[0]
best_fitness = best_individual.fitness.values[0]
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")

# Plot the mean and best fitness values
plt.plot(range(1, n_gen+1), mean_fitness_values, label='Mean Fitness')
plt.plot(range(1, n_gen+1), best_fitness_values, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()
