# loading packages
import numpy as np
import pandas as pd

PREFIX = "n10m20t30seed"
MODEL = ["f2both_prior", "r5graded_multi", "f2pop", "r5ind",
         "r5both", "r5pop", "r5DSEM", "r5sequential_multi", 
         "r5Gaussian", "r5TVAR", "r5gpcm_multi"]

RESULT_PATH = "./log/"
MAX_SEED = 25

results = np.zeros((len(MODEL), MAX_SEED))

for SEED in range(1, MAX_SEED + 1):
    for i in range(len(MODEL)):
        loading_file = RESULT_PATH + "{}{}{}.log".format(PREFIX, SEED, MODEL[i])

        with open(loading_file) as f:
            data = f.readlines()
            data = data[-5]
            data = int(data.split(":")[1].replace(" ", "")[:-5])
            results[i, SEED - 1] = data

results_mu = np.round(np.mean(results, axis=1), decimals=3)
results_std = np.round(np.std(results, axis=1) / np.sqrt(MAX_SEED), decimals=3)

results = pd.DataFrame({"model": MODEL, "avg runtime": results_mu, "std runtime": results_std})
print(results)

        

