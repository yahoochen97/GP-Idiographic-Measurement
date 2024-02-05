# loading packages
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5",\
            "trait_E", "trait_A", "trait_O", "trait_N", "trait_C"]

PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5"]

RESULT_PATH = "./results/GP_ESM/prediction/"

def main(args):

    MODELS = ["pop", "both"]
    MEASURES = ["test_acc", "test_ll"]
    results = np.zeros((len(MODELS), len(PRED_TYPES), len(MEASURES)))

    for j in range(len(PRED_TYPES)):
        for i in range(len(MODELS)):
            cov_file = "{}_{}.npz".format(MODELS[i], PRED_TYPES[j])
            data = np.load(RESULT_PATH + cov_file)

            results[i,j,0] = data["test_acc"]
            results[i,j,1] = data["test_ll"]
 
    results = np.round(results, decimals=3)

    result = pd.DataFrame(results[:,:,0], columns=PRED_TYPES)
    result = result.rename(index=dict(zip([i for i in range(len(MODELS))], MODELS)))
    
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    MODELS = ["IPGP-pop", "IPGP"]
    for i in range(len(MODELS)):
        plt.plot(range(5), results[i,:,0], label=MODELS[i])
    plt.ylim([0.2, 0.5])
    plt.legend(loc=0, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24) 
    plt.xlabel("horizon (days)", fontsize=24)
    plt.tick_params(bottom=False)
    plt.ylabel("predictive acc", fontsize=24)
    plt.savefig(RESULT_PATH+"last_acc.pdf", bbox_inches='tight')

    result = pd.DataFrame(results[:,:,1], columns=PRED_TYPES)
    result = result.rename(index=dict(zip([i for i in range(len(MODELS))], MODELS)))
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    MODELS = ["IPGP-pop", "IPGP"]
    for i in range(len(MODELS)):
        plt.plot(range(5), results[i,:,1], label=MODELS[i])
    plt.ylim([0.2, 0.5])
    plt.legend(loc=0, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24) 
    plt.tick_params(bottom=False)
    plt.xlabel("horizon (days)", fontsize=24)
    plt.ylabel("predictive ll", fontsize=24)
    plt.savefig(RESULT_PATH+"last_ll.pdf", bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    args = vars(parser.parse_args())
    main(args)