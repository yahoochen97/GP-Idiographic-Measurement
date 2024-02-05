# loading packages
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5",\
            "trait_E", "trait_A", "trait_O", "trait_N", "trait_C"]

PRED_TYPES = ["last_2", "last_1", "last_3", "last_4", "last_5"]

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
    
    fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)
    def plot_result(results, MEASURE, i):
        ax = axs[i]
        MODELS = ["IPGP-pop", "IPGP"]
        colors = ["orange", "blue"]
        for j in range(len(MODELS)):
            ax.plot(range(1,6), results[j,:], label=MODELS[j], color=colors[j])
        ax.set_ylim([0.2, 0.5])
        if i==1:
            ax.set_legend(loc=0, fontsize=20)
        XTICKS = [1,2,3,4,5]
        YTICKS = [0.2, 0.3, 0.4, 0.5]
        if MEASURE=="ll":
            YTICKS = [-1.9, -1.7,-1.5,-1.3]
        ax.set_xticks(XTICKS,XTICKS, fontsize=20)
        ax.set_yticks(YTICKS, YTICKS, fontsize=20) 
        ax.set_xlabel("horizon (days)", fontsize=20)
        ax.set_tick_params(left=False, bottom=False)
        ax.set_ylabel("predictive " + MEASURE, fontsize=20)
        ax.grid(axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.savefig(RESULT_PATH + "last_" + "acc_ll" +".pdf", bbox_inches='tight')
    plt.close(fig=fig)

    plot_result(results, "acc", 0)
    plot_result(results, "ll", 1)
   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    args = vars(parser.parse_args())
    main(args)