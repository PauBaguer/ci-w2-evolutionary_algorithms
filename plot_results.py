import json

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

folder = "run2"
k_runs = 3

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

combine_indexes_n0_h2 = np.array([0,4,8])
combine_indexes_n0_h5 = np.array([1,5,9])
combine_indexes_n10_h2 = np.array([2,6,10])
combine_indexes_n10_h5 = np.array([3,7,11])
def average_runs(arr):
    arr = np.array(arr)

    n0_h2 = arr[combine_indexes_n0_h2]
    n0_h5 = arr[combine_indexes_n0_h5]
    n10_h2 = arr[combine_indexes_n10_h2]
    n10_h5 = arr[combine_indexes_n10_h5]

    df = pd.DataFrame({
        'Noise_0_Hardness_2': n0_h2,
        'Noise_0_Hardness_5': n0_h5,
        'Noise_10_Hardness_2': n10_h2,
        'Noise_10_Hardness_5': n10_h5,
    })

    for col in df.columns:
        print(f"MSE: {np.mean(df[col]):3f}+-{stats.sem(df[col]) * 1.967:3f}")

    return df

print("===MLP===")
print("MSE")
MLP_mse_df = average_runs(MLP_mse)
print("Training time")
MLP_time_df = average_runs(MLP_time)

print("===GA===")
print("MSE")
GA_mse_df = average_runs(GA_mse)
print("Training time")
GA_time_df = average_runs(GA_time)

print("===ES===")
print("MSE")
ES_mse_df = average_runs(ES_mse)
print("Training time")
ES_time_df = average_runs(ES_time)





# data_for_boxplot1 = pd.DataFrame({'Adam':MLP_mse, 'GA':GA_mse})#, 'CMA-ES':ES_mse})
# data_for_boxplot2 = pd.DataFrame({'Adam':MLP_time, 'GA':GA_time})#, 'CMA-ES':ES_time})
# labels = ["Adam", "GA", ]#"CMA-ES"]
#
# Create boxplot time predict
#plt.figure(figsize=(5, 4))
fig,ax = plt.subplots(1,2)
fig.subplots_adjust(top=0.99, right=0.99)
ax[0].boxplot(MLP_mse_df[['Noise_0_Hardness_2', 'Noise_0_Hardness_5']], labels=['N=0, H=2', 'N=0, H=5'])
#plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
ax[0].set_xlabel('Datasets')
ax[0].set_ylabel('MSE')
ax[1].boxplot(MLP_mse_df[['Noise_10_Hardness_2', 'Noise_10_Hardness_5']], labels=['N=10, H=2', 'N=10, H=5'])
ax[1].set_xlabel('Datasets')
#plt.title('Boxplot of Model Test MSE')
plt.savefig(f'{folder}/Boxplot_mse_mlp.png')
plt.close()



fig,ax = plt.subplots(1,2)
fig.subplots_adjust(top=0.99, right=0.99)
ax[0].boxplot(GA_mse_df[['Noise_0_Hardness_2', 'Noise_0_Hardness_5']], labels=['N=0, H=2', 'N=0, H=5'])
#plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
ax[0].set_xlabel('Datasets')
ax[0].set_ylabel('MSE')
ax[1].boxplot(GA_mse_df[['Noise_10_Hardness_2', 'Noise_10_Hardness_5']], labels=['N=10, H=2', 'N=10, H=5'])
ax[1].set_xlabel('Datasets')
#plt.title('Boxplot of Model Test MSE')
plt.savefig(f'{folder}/Boxplot_mse_GA.png')
plt.close()


fig,ax = plt.subplots(1,2)
fig.subplots_adjust(top=0.99, right=0.99)
ax[0].boxplot(ES_mse_df[['Noise_0_Hardness_2', 'Noise_0_Hardness_5']], labels=['N=0, H=2', 'N=0, H=5'])
#plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
ax[0].set_xlabel('Datasets')
ax[0].set_ylabel('MSE')
ax[1].boxplot(ES_mse_df[['Noise_10_Hardness_2', 'Noise_10_Hardness_5']], labels=['N=10, H=2', 'N=10, H=5'])
ax[1].set_xlabel('Datasets')
#plt.title('Boxplot of Model Test MSE')
plt.savefig(f'{folder}/Boxplot_mse_ES.png')
plt.close()





#
# plt.figure(figsize=(5, 4))
# #plt.subplots_adjust(bottom=0.345, left=0.24, top=0.92,right=0.99)
# plt.boxplot(data_for_boxplot2, labels=labels)
# #plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# #plt.xlabel('Models')
# plt.ylabel('Training time [s]')
# plt.title('Boxplot of Training time')
# plt.savefig(f'{folder}/Boxplot_training_time.png')
# plt.close()
#
# print()
