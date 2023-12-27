import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


MLP_file_path = "MLP_runs.json"
GA_file_path = "GA_runs.json"
ES_file_path = "ES_runs.json"

MLP_f = open(MLP_file_path)
GA_f = open(GA_file_path)
ES_f = open(ES_file_path)

MLP_data = json.load(MLP_f)
GA_data = json.load(GA_f)
ES_data = json.load(ES_f)


print(MLP_data)
print()

MLP_mse = [run['mse']for run in MLP_data.values()]
MLP_time = [run['time']for run in MLP_data.values()]
MLP_n_neurons = [run['time']for run in MLP_data.values()]

GA_mse = [run['mse']for run in GA_data.values()]
GA_time = [run['time']for run in GA_data.values()]
GA_n_neurons = [run['time']for run in GA_data.values()]

ES_mse = [run['mse']for run in ES_data.values()]
ES_time = [run['time']for run in ES_data.values()]
ES_n_neurons = [run['time']for run in ES_data.values()]

print(f"MLP MSE: {np.mean(MLP_mse):3f}+-{stats.sem(MLP_mse) * 1.967:3f}")
print(f"MLP training time: {np.mean(MLP_time):3f}+-{stats.sem(MLP_time) * 1.967:3f}")

print(f"GA MSE: {np.mean(GA_mse):3f}+-{stats.sem(GA_mse) * 1.967:3f}")
print(f"GA training time: {np.mean(GA_time):3f}+-{stats.sem(GA_time) * 1.967:3f}")

print(f"ES MSE: {np.mean(ES_mse):3f}+-{stats.sem(ES_mse) * 1.967:3f}")
print(f"ES training time: {np.mean(ES_time):3f}+-{stats.sem(ES_time) * 1.967:3f}")


print()
