import json

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

folder = "run2"

MLP_file_path = f"{folder}/MLP_runs.json"
GA_file_path = f"{folder}/GA_runs.json"
ES_file_path = f"{folder}/ES_runs.json"

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

data_for_boxplot1 = pd.DataFrame({'Adam':MLP_mse, 'GA':GA_mse, 'CMA-ES':ES_mse})
data_for_boxplot2 = pd.DataFrame({'Adam':MLP_time, 'GA':GA_time, 'CMA-ES':ES_time})
labels = ["Adam", "GA", "CMA-ES"]

# Create boxplot time predict
plt.figure(figsize=(5, 4))
#plt.subplots_adjust(bottom=0.345, left=0.24, top=0.92,right=0.99)
plt.boxplot(data_for_boxplot1, labels=labels)
#plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
#plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('Boxplot of Model Test MSE')
plt.savefig(f'{folder}/Boxplot_mse.png')
plt.close()

plt.figure(figsize=(5, 4))
#plt.subplots_adjust(bottom=0.345, left=0.24, top=0.92,right=0.99)
plt.boxplot(data_for_boxplot2, labels=labels)
#plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
#plt.xlabel('Models')
plt.ylabel('Training time [s]')
plt.title('Boxplot of Training time')
plt.savefig(f'{folder}/Boxplot_training_time.png')
plt.close()

print()
