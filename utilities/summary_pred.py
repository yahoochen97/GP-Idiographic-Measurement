# loading packages
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5",\
            "trait_E", "trait_A", "trait_O", "trait_N", "trait_C"]

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
            results[i,j,1] = -data["test_ll"]
 
    results = np.round(results, decimals=3)
    print(results[:,5:10,0])
    print(-results[:,5:10,1])
    
    fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
    
    MEASURES = ["acc", "nll"]
    for i in range(len(MEASURES)):
        for k in range(2):
            MEASURE = MEASURES[i]
            ax = axs[i,k]
            MODELS = ["IPGP-pop", "IPGP"]
            colors = ["orange", "blue"]
            for j in range(len(MODELS)):
                ax.plot(range(1,6), results[j,(k*5):(k*5+5)], label=MODELS[j], color=colors[j])
            
            XTICKS = [1,2,3,4,5]
            ax.set_xticks(XTICKS, XTICKS)
            if k==1:
                ax.set_xticks(XTICKS, ["E", "A", "O", "N", "C"])
            YTICKS = [0.2, 0.3, 0.4, 0.5]
            if MEASURE=="nll":
                YTICKS = [1.2, 1.4,1.6,1.8]
                ax.set_ylim([0.2, 0.5])
            else:
                ax.set_ylim([0.2, 0.5])
            
            ax.set_yticks(YTICKS, YTICKS)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20) 
            if k==0:
                ax.set_xlabel("horizon (days)", fontsize=20)
            else:
                ax.set_xlabel("traits (Big Five)", fontsize=20)
            ax.tick_params(left=False, bottom=False)
            if i==0:
                ax.set_ylabel("predictive " + MEASURE, fontsize=20)

            if i==1:
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-') for c in colors]
                ax.legend(lines, MODELS, loc=0, fontsize=12)
                ax.set_ylim([1.3,1.9])
            else:
                ax.get_xaxis().set_visible(False)
            ax.grid(axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.savefig(RESULT_PATH + "last_" + "acc_ll" +".pdf", bbox_inches='tight')
    plt.close(fig=fig)
   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    args = vars(parser.parse_args())
    main(args)