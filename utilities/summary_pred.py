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
    
    def plot_result(results, TASK, MEASURE):
        fig, ax = plt.subplot(figsize=(6, 5))
        MODELS = ["IPGP-pop", "IPGP"]
        for i in range(len(MODELS)):
            plt.plot(range(1,6), results[i,:], label=MODELS[i])
        plt.ylim([0.2, 0.5])
        plt.legend(loc=0, fontsize=20)
        XTICKS = [1,2,3,4,5]
        YTICKS = [0.2, 0.3, 0.4, 0.5]
        if MEASURE=="ll":
            YTICKS = [-1.9, -1.7,-1.5,-1.3]
        plt.xticks(XTICKS,XTICKS, fontsize=20)
        plt.yticks(YTICKS, YTICKS, fontsize=20) 
        plt.xlabel("horizon (days)", fontsize=20)
        plt.tick_params(bottom=False)
        plt.ylabel("predictive " + MEASURE, fontsize=20)
        ax.grid(axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig(RESULT_PATH+TASK + "_" + MEASURE +".pdf", bbox_inches='tight')
        plt.close(fig=fig)

    plot_result(results, "last", "acc")
    plot_result(results, "last", "ll")
   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    args = vars(parser.parse_args())
    main(args)