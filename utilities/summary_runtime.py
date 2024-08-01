# loading packages
import numpy as np
import pandas as pd

PREFIX = "n10m20t30seed"
MODEL = ["f2both_prior", "r5graded_multi", "f2pop", "r5ind",
         "r5both", "r5pop", "r5DSEM", "r5sequential_multi", 
         "r5Gaussian", "r5TVAR", "r5gpcm_multi"]

RESULT_PATH = "./log"
MAX_SEED = 1

results = np.zeros((len(MODEL), MAX_SEED))

for SEED in range(1, MAX_SEED + 1):
    for i in range(MODEL):
        loading_file = "{}{}{}.log".format(PREFIX, SEED, MODEL)

        with open(loading_file) as f:
            data = f.readlines()
            data = data[-5]
            data = data.split(":")[1].replace(" ", "")
            print(data)

        

