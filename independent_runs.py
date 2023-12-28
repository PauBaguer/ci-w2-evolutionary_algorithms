
import numpy as np
import json
import ES_FINAL, GA_FINAL, MLP

def main():
    folder = "run3"
    k_runs = 3
    dataset_sample_size = 2000
    dataset_noise_level = 0.25
    ###################
    #      MLP        #
    ###################
    MLP_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = MLP.do_MLP(k, folder, dataset_sample_size, dataset_noise_level)
        MLP_runs[k] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons)}

    with open(f"{folder}/MLP_runs.json", "w") as js:
        json.dump(MLP_runs, js)

    ###################
    #        GA       #
    ###################

    GA_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = GA_FINAL.do_GA(k, folder, dataset_sample_size, dataset_noise_level)
        GA_runs[k] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons)}

    with open(f"{folder}/GA_runs.json", "w") as js:
        json.dump(GA_runs, js)


    ###################
    #        ES       #
    ###################

    ES_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = ES_FINAL.do_ES(k, folder, dataset_sample_size, dataset_noise_level)
        ES_runs[k] = {'mse':mse, 'time':time, 'n_neurons':float(n_neurons)}

    with open(f"{folder}/ES_runs.json", "w") as js:
        json.dump(ES_runs, js)



if __name__ == "__main__":
    main()