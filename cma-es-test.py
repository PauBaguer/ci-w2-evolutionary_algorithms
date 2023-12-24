import numpy as np
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad
import keras
import visualkeras
import tensorflow as tf

from cma import CMA


max_epochs = 500
def fitness_func(solution):
    global data_inputs, data_outputs, keras_ga, model
    solution_np=solution.numpy()
    sol = []
    split1 = np.array(solution_np[0,:15])
    split2 = np.array(solution_np[0,15:20])
    split3 = np.array(solution_np[0,20:25])
    split4 = np.array(solution_np[0, 25:26])
    sol.append(split1.reshape((3,5)))
    sol.append(split2.reshape((5,)))
    sol.append(split3.reshape((5,1)))
    sol.append(split4)
    #model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
     #                                                            weights_vector=solution)

    model.set_weights(weights=sol)

    predictions = model.predict(data_inputs)

    mae = tensorflow.keras.losses.MeanAbsoluteError()
    abs_error = mae(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error

    solution_fitness_t = tf.convert_to_tensor([solution_fitness])

    return solution_fitness_t

def logging_function(cma, logger):
    if cma.generation % 10 == 0:
        fitness = cma.best_fitness()
        logger.info(f'Generation {cma.generation} - fitness {fitness}')

    if cma.termination_criterion_met or cma.generation == max_epochs:
        sol = cma.best_solution()
        fitness = cma.best_fitness()
        logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')

input_layer  = tensorflow.keras.layers.Input(3)
dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

weights = model.weights

weights_vector = pygad.kerasga.model_weights_as_vector(model=model)

keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=1)

# Data inputs
data_inputs = numpy.array([[0.02, 0.1, 0.15],
                           [0.7, 0.6, 0.8],
                           [1.5, 1.2, 1.7],
                           [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = numpy.array([[0.1],
                            [0.6],
                            [1.3],
                            [2.5]])

num_generations = 250
num_parents_mating = 5
initial_population = keras_ga.population_weights


# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        initial_population=initial_population,
#                        fitness_func=fitness_func,
#                        on_generation=callback_generation)

cma = CMA(
    initial_solution=initial_population[0],
    initial_step_size=0.000001,
    fitness_function=fitness_func,
    callback_function=logging_function,
)

best_solution, best_fitness = cma.search(max_epochs)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")