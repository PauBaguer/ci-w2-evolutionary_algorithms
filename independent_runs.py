
import numpy as np
import json
import ES_FINAL, GA_FINAL
import MLP


def main():
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
                mse, time, n_neurons = MLP.do_MLP(k, n, h)
                MLP_runs[count] = {'mse': mse, 'time': time, 'n_neurons': int(n_neurons)}
                count += 1

    with open("MLP_runs.json", "w") as js:
        json.dump(MLP_runs, js)

    ###################
    #        GA       #
    ###################

    GA_runs = {}
    count = 0
    for k in range(k_runs):
        for n in noises:
            for h in hardness:
                mse, time, n_neurons, wd = GA_FINAL.do_GA(k, n, h)
                GA_runs[count] = {'mse': mse, 'time': time, 'n_neurons': int(n_neurons), 'weight_decay': wd}
                count += 1

    with open("GA_runs.json", "w") as js:
        json.dump(GA_runs, js)


    ###################
    #        ES       #
    ###################

    ES_runs = {}
    count = 0
    for k in range(k_runs):
        for n in noises:
            for h in hardness:
                mse, time, n_neurons, wd = ES_FINAL.do_ES(k, n, h)
                ES_runs[count] = {'mse':mse, 'time':time, 'n_neurons':int(n_neurons), 'weight_decay':wd}
                count += 1

    with open("ES_runs.json", "w") as js:
        json.dump(ES_runs, js)



if __name__ == "__main__":
    main()