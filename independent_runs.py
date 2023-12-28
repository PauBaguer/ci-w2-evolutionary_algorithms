
import numpy as np
import json
import ES_FINAL, GA_FINAL, MLP

def main():
    folder = "run2"
    k_runs = 3
    noises = [0, 10]
    hardness = [2, 5]

    ###################
    #      MLP        #
    ###################
    count = 0
    MLP_runs = {}
    for k in range(k_runs):
        for n in noises:
            for h in hardness:
                mse, time, n_neurons = MLP.do_MLP(k, n, h, folder)
                MLP_runs[count] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons), 'weight_decay': 10e-4}
                count += 1

    with open(f"{folder}/MLP_runs.json", "w") as js:
        json.dump(MLP_runs, js)

    ###################
    #        GA       #
    ###################

    GA_runs = {}
    count = 0
    for k in range(k_runs):
        for n in noises:
            for h in hardness:
                mse, time, n_neurons, wd = GA_FINAL.do_GA(k, n, h, folder)
                GA_runs[count] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons), 'weight_decay': wd}
                count += 1

    with open(f"{folder}/GA_runs.json", "w") as js:
        json.dump(GA_runs, js)


    ###################
    #        ES       #
    ###################

    ES_runs = {}
    count = 0
    for k in range(k_runs):
        for n in noises:
            for h in hardness:
                mse, time, n_neurons, wd = ES_FINAL.do_ES(k, n, h, folder)
                ES_runs[count] = {'mse':mse, 'time':time, 'n_neurons':float(n_neurons), 'weight_decay':wd}
                count += 1

    with open(f"{folder}/ES_runs.json", "w") as js:
        json.dump(ES_runs, js)



if __name__ == "__main__":
    main()