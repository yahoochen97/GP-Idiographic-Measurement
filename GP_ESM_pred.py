# loading packages
import numpy as np
import torch
import os
import argparse
import pandas as pd
import matplotlib.pylab as plt
from gpytorch.mlls import VariationalELBO

from scipy.stats import norm
torch.manual_seed(8927)
np.random.seed(8927)
torch.set_default_dtype(torch.float64)

import warnings
warnings.filterwarnings("ignore")

from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
from utilities.util import OrdinalLMC, OrdinalLikelihood
from utilities.util import correlation_matrix_distance, evaluate_gpr

def main(args):
    SEED = int(args["seed"])
    pred_type = args["pred_type"]
    load_batch_size = 512
    num_inducing = 5000
    model_type = args["model_type"]
    print("loading data...")
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    Items_loopr = data.columns.to_list()

    # set random seed
    torch.manual_seed(8927+SEED)
    np.random.seed(8927+SEED) 

    # rename volat to violat
    for i in range(1,5):
        Items_loopr[Items_loopr.index("Volat.{}".format(i))] = "Violat.{}".format(i)

    # generate item map from original to current using ESM codebook
    codebook = pd.read_excel("./data/ESM_Codebook.xlsx")
    item_mapping = dict(zip([x.replace(" ", "") for x in codebook.iloc[:,0].to_list()],\
                            [x.replace(" ", "") for x in codebook.iloc[:,1].to_list()]))
    reverse_code = codebook.iloc[:,2].to_list()

    # read data
    data = pd.read_csv("./data/GP_ESM_cleaned.csv")

    data.columns = [x.replace(" ", "") for x in data.columns]
    ESM_items = [x.replace(" ", "") for x in codebook.iloc[:,0].to_list() if x.replace(" ", "") in Items_loopr]
    reverse_code = [reverse_code[i] for i in range(codebook.shape[0]) if codebook.iloc[i,0].replace(" ", "") in Items_loopr]
    reverse_code = np.array(reverse_code).reshape(-1,1)
    time_diff = (pd.to_datetime(data.RecordedDate, format='%Y-%m-%d %H:%M:%S')-pd.to_datetime(data.RecordedDate[0])).dt
    data["day"] = time_diff.days
    data["day"] += time_diff.seconds/60/60/25

    n = data.PID.unique().shape[0]
    m = len(ESM_items)
    horizon = data.day.max()

    # transform to row data frame
    train_x = torch.zeros((n*m*data.n.max(),3))
    train_y = torch.zeros((n*m*data.n.max(),))

    ITER = 0
    for iter in range(data.shape[0]):
        for j in range(m):
            train_x[ITER, 0] = data.PID[iter]
            train_x[ITER, 1] = j
            train_x[ITER, 2] = data.day[iter]
            train_y[ITER] = data[item_mapping[ESM_items[j]]][iter]
            if reverse_code[j,0]==1:
                train_y[ITER] = 6 - train_y[ITER]
            ITER += 1

    train_x = train_x[~train_y.isnan()]
    train_y = train_y[~train_y.isnan()]
    train_x = train_x[train_y!=0]
    train_y = train_y[train_y!=0]

    # split train/test for forecasting
    parts = pred_type.split("_")
    if parts[0]=="last":
        test_mask = (horizon-int(parts[1]))<=train_x[:,2]
        test_mask = test_mask & (train_x[:,2]<(horizon-int(parts[1])+1))
        test_x = train_x[test_mask]
        test_y = train_y[test_mask]
        train_mask = train_x[:,2]<=(horizon-7)
        train_y = train_y[train_mask]
        train_x = train_x[train_mask]
    elif parts[0]=="trait":
        TS = ["E","A","C","N","O"]
        idx = TS.index(parts[1])
        test_mask = (idx*9<=train_x[:,1])
        test_mask = test_mask & (train_x[:,1]<(idx*9+9))
        # test_mask = torch.bernoulli(test_mask*0.5)
        test_x = train_x[test_mask]
        test_y = train_y[test_mask]
        train_y = train_y[~test_mask]
        train_x = train_x[~test_mask]

    # build data loader
    C = 5
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=load_batch_size, shuffle=False)

    # initialize likelihood and model
    inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    likelihood = OrdinalLikelihood(thresholds=torch.tensor([-20.,-2.,-1.,1.,2.,20.]))
    pop_rank = 5
    unit_rank = 1
    model = OrdinalLMC(inducing_points,n=n,m=m,C=C,horizon=horizon,\
                    pop_rank=pop_rank, unit_rank=unit_rank, model_type=model_type)

    model.train()
    likelihood.train()

    # initialize covariance of pop factors
    pop_prior = np.load("./results/GP_ESM/both_f1_Feb.npz")
    model.pop_task_covar_module.covar_factor.data = torch.tensor(pop_prior["pop_factor"])
    
    if model_type=="both":
        pop_kernel =  model.pop_task_covar_module.covar_matrix.evaluate()
        for i in range(n):
            unit_cov =  torch.tensor(pop_prior["unit_{}_covariance".format(i)])
            unit_cov = pop_kernel - unit_cov
            U, S, V = torch.pca_lowrank(unit_cov.double(), q = 1)
            X_init = U.matmul(S).reshape((-1,1)) @ V.T
            model.unit_task_covar_module[i].covar_factor.data = X_init
            model.unit_task_covar_module[i].raw_var.data = torch.tensor([-10.])

    # Our loss object. We're using the VariationalELBO
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # prediction
    model.eval()
    likelihood.eval()
    print("start predicting...")

    # train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader)
    test_acc, test_ll = evaluate_gpr(model, likelihood, test_loader, mll)

    results = {}
    print("out-of-sample evaluatiion...")
    print("test acc: {}".format(test_acc))
    print("test ll: {}".format(test_ll))

    log_lik = test_ll * test_x.size(0)
    results["test_acc"] = test_acc
    results["test_ll"] = test_ll
    results["log_lik"] = log_lik

    PATH = "./results/GP_ESM/prediction/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    np.savez(PATH+"{}_{}.npz".format(model_type, pred_type), **results)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-k model_type -p pred_type -f factor')
    parser.add_argument('-p','--pred_type', help='type of prediction', required=False)
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-s','--seed', help='random seed', required=False)
    args = vars(parser.parse_args())
    main(args)