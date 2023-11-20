# loading packages
import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import pyreadr

from util import correlation_matrix_distance, plot_task_kernel

MODELS = ["pop", "ind", "both", "graded_uni", "graded_multi", "gpcm_uni",\
           "gpcm_multi", "sequential_uni", "sequential_multi"]

def main(args):
    MAXSEED = int(args["maxseed"])
    n = int(args["num_unit"])
    m = int(args["num_item"])
    horizon = int(args["num_period"])
    RANK = int(args["rank"])

    MEASURES = ["train_acc", "train_ll", "test_acc", "test_ll"] + ["cov_dist_{}".format(i) for i in range(n)]
    results = np.zeros((len(MODELS), len(MEASURES), MAXSEED))

    for SEED in range(MAXSEED):
        SEED_ = SEED + 1
        loading_file = "loadings_n{}_m{}_t{}_rank{}_SEED{}.npz".format(n,m,horizon,RANK,SEED_)
        DGP_PATH = "./data/synthetic/"
        dgp_loadings = np.load(DGP_PATH + loading_file)
        dgp_pop_loadings = dgp_loadings["pop_loadings"]
        dgp_covariance = dgp_pop_loadings.T @ dgp_pop_loadings
        dgp_unit_loadings = dgp_loadings["unit_loadings"]
        for i in range(len(MODELS)):
            if i<3:
                PATH = "./results/synthetic/"
                cov_file = "cov_{}_n{}_m{}_t{}_rank{}_SEED{}.npz".format(MODELS[i], n,m,horizon,RANK,SEED_)
                data = np.load(PATH + cov_file)
            else:
                cov_file = "{}_n{}_m{}_t{}_rank{}_SEED{}.RData".format(MODELS[i], n,m,horizon,RANK,SEED_)
                data = pyreadr.read_r(PATH + cov_file)
            
            results[i,0,SEED] = data["train_acc"]
            results[i,1,SEED] = data["train_ll"]
            results[i,2,SEED] = data["test_acc"]
            results[i,3,SEED] = data["test_ll"]
            for unit_i in range(n):
                dgp_covariance = dgp_pop_loadings.T @ dgp_pop_loadings + dgp_unit_loadings[i].T @ dgp_unit_loadings[i]
                if i<3:
                    unit_cov = data["unit_{}_covariance".format(unit_i)]
                else:
                    unit_cov = data["correlation_matrix"]
                unit_dist = correlation_matrix_distance(dgp_covariance, unit_cov)
                results[i,4+unit_i,SEED] = unit_dist


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-n num_unit -m num_item -t num_period -s seed')
    parser.add_argument('-n','--num_unit', help='number of units', required=False)
    parser.add_argument('-m','--num_item', help='number of items', required=False)
    parser.add_argument('-t','--num_period', help='number of periods', required=False)
    parser.add_argument('-s','--maxseed', help='max seed', required=False)
    parser.add_argument('-r','--rank', help='rank of item correlation matrix', required=False)
    args = vars(parser.parse_args())
    main(args)