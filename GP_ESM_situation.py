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
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader
from utilities.util import OrdinalLMC, OrdinalLikelihood
from utilities.util import correlation_matrix_distance, plot_task_kernel
from utilities.util import plot_agg_task_kernel, evaluate_gpr

# add situation data
num_situation = 1
situations = ['Di_Sit_' + str(i) for i in range(1,num_situation+1)]

def main(args):
    load_batch_size = 512
    num_inducing = 5000
    num_epochs = 5
    FACTOR = int(args["factor"])
    model_type = args["model_type"]
    print("loading data...")
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    Items_loopr = data.columns.to_list()

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
    time_diff = (pd.to_datetime(data.RecordedDate, format='%Y-%m-%d %H:%M:%S')-\
                 pd.to_datetime(data.RecordedDate.iloc[0])).dt
    data["day"] = time_diff.days
    data["day"] += time_diff.seconds/60/60/25

    PIDs = data.PID.unique()
    n = PIDs.shape[0]
    PID_mapping = dict(zip(PIDs, range(n)))
    m = len(ESM_items)
    horizon = data.day.max()

    # transform to row data frame
    train_x = torch.zeros((n*m*data.n.max(),3 + len(situations)))
    train_y = torch.zeros((n*m*data.n.max(),))

    ITER = 0
    for iter in range(data.shape[0]):
        for j in range(m):
            train_x[ITER, 0] = PID_mapping[data.PID.iloc[iter]]
            train_x[ITER, 1] = j
            train_x[ITER, 2] = data.day.iloc[iter]
            for k in range(len(situations)):
                train_x[ITER, 3+k] = data.iloc[iter][situations[k]] - 1

            train_y[ITER] = data[item_mapping[ESM_items[j]]].iloc[iter]
            if reverse_code[j,0]==1:
                train_y[ITER] = 6 - train_y[ITER]
            ITER += 1

    train_x = train_x[~train_y.isnan()]
    train_y = train_y[~train_y.isnan()]
    train_x = train_x[train_y!=0]
    train_y = train_y[train_y!=0]
    mask = ~torch.any(train_x.isnan(),dim=1)
    train_x = train_x[mask]
    train_y = train_y[mask]

    print(train_x.shape)

    # build data loader
    C = 5
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)

    # initialize likelihood and model
    inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    likelihood = OrdinalLikelihood(thresholds=torch.tensor([-20.,-2.5,-1.,1.,2.5,20.]))
    pop_rank = 5
    unit_rank = FACTOR
    model = OrdinalLMC(inducing_points,n=n,m=m,C=C,horizon=horizon, d=len(situations),\
                    pop_rank=pop_rank, unit_rank=unit_rank, model_type=model_type)

    model.train()
    likelihood.train()

    # initialize covariance of pop factors
    pop_prior = np.load("./results/loopr/loopr_pop_f{}_e10.npz".format(pop_rank))
    loopr_idx = [Items_loopr.index(x) for x in ESM_items]
    model.pop_task_covar_module.covar_factor.data = torch.tensor(pop_prior["pop_factor"][loopr_idx])
    model.pop_task_covar_module.covar_factor.requires_grad = False

    # select hyperparameters to learn
    for i in range(n):
        model.t_covar_module[i].lengthscale = 7 # data.day.max() // 3
    model.fixed_module.raw_lengthscale.requires_grad = False

    final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale, model.pop_task_covar_module.covar_factor}) + \
                    list(likelihood.parameters())

    optimizer = torch.optim.Adam(final_params, lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # train GPR
    print("start training...")
    for i in range(num_epochs):
        log_lik = 0
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            log_lik += -loss.item()*y_batch.shape[0]
            if j % 50 == 0:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    # prediction
    model.eval()
    likelihood.eval()
    print("start predicting...")

    train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader, mll)

    results = {}
    print("in-sample evaluatiion...")
    print("train acc: {}".format(train_acc))
    print("train ll: {}".format(train_ll))

    log_lik = train_ll * train_x.size(0)
    results["train_acc"] = train_acc
    results["train_ll"] = train_ll
    results["log_lik"] = log_lik

    task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
    results["pop_covariance"] = task_kernel
    results["pop_factor"] = model.pop_task_covar_module.covar_factor.data.detach().numpy()

    unit_covariance = np.zeros((n,m,m))
    for i in range(n):
        task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        for k in range(len(situations)):
            task_kernel += model.situation_task_covar_module[k].covar_matrix.evaluate().detach().numpy() \
                        * train_x[i,(3+k)].numpy()
        if model_type!="pop":
            task_kernel += model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy()
        unit_covariance[i] = task_kernel
        results["unit_{}_covariance".format(i)] = task_kernel
        results["unit_{}_factor".format(i)] = model.unit_task_covar_module[i].covar_factor.detach().numpy()

    for k in range(len(situations)):
        results["situation_{}_factor".format(k)] = model.situation_task_covar_module[k].covar_factor.detach().numpy()
        results["situation_{}_covar".format(k)] = model.situation_task_covar_module[k].covar_matrix.evaluate().detach().numpy()

    PATH = "./results/GP_ESM_2/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    np.savez(PATH+"situation_{}_f{}.npz".format(model_type, FACTOR), **results)


def plot_situation():
    PATH = "./results/GP_ESM_2/"
    results = np.load(PATH+"situation_pop_f5.npz") 

    print(results["train_acc"])
    print(results["train_ll"])

    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    Items_loopr = data.columns.to_list()

    # rename volat to violat
    for i in range(1,5):
        Items_loopr[Items_loopr.index("Volat.{}".format(i))] = "Violat.{}".format(i)

    # generate item map from original to current using ESM codebook
    codebook = pd.read_excel("./data/ESM_Codebook.xlsx")
    ESM_items = [x.replace(" ", "") for x in codebook.iloc[:,0].to_list() if x.replace(" ", "") in Items_loopr]
    data = pd.read_csv("./data/GP_ESM_cleaned.csv")
    n = data.PID.unique().shape[0]
    pop_task_kernel = results["pop_covariance"]

    # plot individual kernel
    all_cov = np.zeros((len(situations),len(ESM_items),len(ESM_items)))
    for k in range(len(situations)):
        ind_task_kernel = results["situation_{}_factor".format(k)]
        all_cov[k] = ind_task_kernel @ ind_task_kernel.transpose() + pop_task_kernel

    # plot_agg_task_kernel(pop_task_kernel + pop_task_kernel, pop_task_kernel, PATH + "pop_{}.pdf".format(k))
    # plot centroids
    for k in range(len(situations)):
        plot_agg_task_kernel(all_cov[k], pop_task_kernel, PATH + "residual_{}.pdf".format(k))
        plot_task_kernel(all_cov[k], \
             np.array(ESM_items), \
             PATH + "all_cov_{}_5.pdf".format(k), SORT=False)
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-k model_type -f factor')
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-f','--factor', help='number of coregionalization factors', required=False)
    args = vars(parser.parse_args())
    main(args)
    # plot_situation()
