# loading packages
import numpy as np
import pandas as pd
import argparse

MODELS = ["pop", "both"]
PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5",\
            "trait_E", "trait_A", "trait_O", "trait_N", "trait_C"]

PRED_TYPES = ["last_1", "last_2", "last_3", "last_4", "last_5"]

RESULT_PATH = "./results/GP_ESM/prediction/"

def main(args):

    MEASURES = ["test_acc", "test_ll"]
    results = np.zeros((len(MODELS), len(PRED_TYPES), len(MEASURES)))

    for j in range(len(PRED_TYPES)):
        for i in range(len(MODELS)):
            cov_file = "{}_{}.npz".format(MODELS[i], PRED_TYPES[i])
            data = np.load(RESULT_PATH + cov_file)

            results[i,j,0] = np.array(data["test_acc"])
            results[i,j,1] = np.array(data["test_ll"])
 
    results = np.round(results, decimals=3)

    results = pd.DataFrame(results[:,:,0], columns=MODELS)
    results = results.rename(index=dict(zip([i for i in range(len(PRED_TYPES))], PRED_TYPES)))
    print(results)

    results = pd.DataFrame(results[:,:,1], columns=PRED_TYPES)
    results = results.rename(index=dict(zip([i for i in range(len(MODELS))], MODELS)))
    print(results)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    args = vars(parser.parse_args())
    main(args)