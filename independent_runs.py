
import numpy as np
import json
import ES_FINAL, GA_FINAL
import MLP


def main():
    k_runs = 5

    ###################
    #      MLP        #
    ###################
    MLP_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = MLP.do_MLP(k)
        MLP_runs[k] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons)}

    with open("MLP_runs.json", "w") as js:
        json.dump(MLP_runs, js)

    ###################
    #        GA       #
    ###################

    GA_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = GA_FINAL.do_GA(k)
        GA_runs[k] = {'mse': mse, 'time': time, 'n_neurons': float(n_neurons)}

    with open("GA_runs.json", "w") as js:
        json.dump(GA_runs, js)


    ###################
    #        ES       #
    ###################

    ES_runs = {}
    for k in range(k_runs):
        mse, time, n_neurons = ES_FINAL.do_ES(k)
        ES_runs[k] = {'mse':mse, 'time':time, 'n_neurons':float(n_neurons)}

    with open("ES_runs.json", "w") as js:
        json.dump(ES_runs, js)



if __name__ == "__main__":
    main()