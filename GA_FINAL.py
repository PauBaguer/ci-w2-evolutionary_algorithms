import numpy as np
import data
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import time

tf.config.run_functions_eagerly(True)

@tf.function
def obtain_mse(individual, X, y):
    n_neurons = int(individual[0])
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
    n_neurons = int(individual[0])
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
    mse = mean_squared_error(y, predictions) + wd * np.sum(np.square(weights))
    return mse








def do_GA(k, n, h, folder):
    global X_train, X_val, X_test, y_train, y_val, y_test

    global max_neurons
    global max_epochs
    global ga
    X_train, X_val, X_test, y_train, y_val, y_test = data.generate_synthetic_data_reg(10000, noise_level=n, hardness=h)
    max_neurons = 100
    max_epochs = 150
    n_weights = X_train.shape[1] * max_neurons + max_neurons

    # Create a binary individual with two genes
    indv_template = BinaryIndividual(ranges=[(1, max_neurons), (1e-10, 1)] + [[-10, 10]] * n_weights, eps=0.001)

    # Create GA engine
    ga = GAEngine(population=Population(indv_template).init(), selection=TournamentSelection(),
                  crossover=UniformCrossover(pc=0.8, pe=0.5), mutation=FlipBitMutation(pm=0.2))
    history = {
        'best_fitness_train': [],
        'best_fitness_val': [],
        'best_mse_train': [],
        'best_mse_val': []
    }

    @ga.fitness_register
    @ga.minimize
    def fitness_function(individual):
        global X_train, y_train
        return float(obtain_fitness(individual.solution, X_train, y_train))

    @ga.analysis_register
    class ConsoleOutput(OnTheFlyAnalysis):
        master_only = True
        interval = 1

        def register_step(self, g, population, engine):
            best_indv = population.best_indv(engine.fitness).solution
            if g % 10 == 0:
                print('Generation: {}, best fitness: {:.3f}'.format(g, -engine.fmax))
            if g == max_epochs:
                print(f'Final solution at gen {g} - (fitness: {-engine.fmax})')
            history['best_fitness_train'].append(-engine.fmax)
            history['best_fitness_val'].append(obtain_fitness(best_indv, X_val, y_val))
            history['best_mse_train'].append(obtain_mse(best_indv, X_train, y_train))
            history['best_mse_val'].append(obtain_mse(best_indv, X_val, y_val))





    ga.fmax = 1e-6  # Set the fitness threshold for early stopping if needed
    start_time = time.time()
    ga.run(ng=max_epochs)
    end_time = time.time()
    train_time = end_time-start_time

    # Retrieve the best individual and its fitness value
    best_individual = ga.population.best_indv(ga.fitness).solution

    # Print the best individual and its fitness value
    print(f"Best individual: {best_individual}, Best fitness: {-ga.fmax}")


    plt.plot(history['best_fitness_train'])
    plt.plot(history['best_fitness_val'])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(['Training', 'Validation'])
    plt.grid(True)
    plt.savefig(f'{folder}/Fitness/GA_training_fitness_{k}_{n}_{h}.png')
    plt.clf()


    plt.plot(history['best_mse_train'])
    plt.plot(history['best_mse_val'])
    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.legend(['Training', 'Validation'])
    plt.grid(True)
    plt.savefig(f'{folder}/MSE/GA_training_mse_{k}_{n}_{h}.png')
    plt.clf()
    mse = obtain_mse(best_individual, X_test, y_test)
    print('MSE on test set:', mse)
    n_neurons = int(best_individual[0])
    wd = best_individual[1]

    return mse, train_time, n_neurons, wd
