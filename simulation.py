# loading packages
import numpy as np
import torch
import os
import pandas as pd
import argparse
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
from utilities.util import correlation_matrix_distance, plot_task_kernel, evaluate_gpr

def main(args):
    SEED = int(args["seed"])
    n = int(args["num_unit"])
    m = int(args["num_item"])
    horizon = int(args["num_period"])
    RANK = int(args["rank"])
    FACTOR = int(args["factor"])
    model_type = args["model_type"]
    load_batch_size = 256
    num_inducing = 1000
    num_epochs = 1

    # load data
    print("loading data...")
    data_file = "./data/synthetic/data_n{}_m{}_t{}_rank{}_SEED{}.csv".format(n,m,horizon,RANK,SEED)
    data = pd.read_csv(data_file)
    train_x = torch.tensor(data[["unit","item","time"]].to_numpy())
    train_y = torch.tensor(data.y)
    C = train_y.unique().size(0)

    # split train/test 
    train_mask = data.train.to_numpy()
    test_x = train_x[train_mask==0]
    test_y = train_y[train_mask==0]
    train_x = train_x[train_mask==1]
    train_y = train_y[train_mask==1]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=load_batch_size, shuffle=False)

    # initialize likelihood and model
    inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    likelihood = OrdinalLikelihood(thresholds=torch.tensor([-20.,\
                                   -2.,-1.,1.,2.,20.]))
    
    pop_rank = FACTOR
    unit_rank = 1
    if model_type=="ind":
        unit_rank = FACTOR
    model = OrdinalLMC(inducing_points,n,m,C,horizon, pop_rank, unit_rank, model_type)

    model.train()
    likelihood.train()

    # select hyperparameters to learn
    for i in range(n):
        model.t_covar_module[i].lengthscale = horizon // 3 if horizon > 1 else 1
    model.fixed_module.raw_lengthscale.requires_grad = False

    if model_type=="both":
        PATH = "./results/synthetic/"
        cov_file = "cov_pop_n{}_m{}_t{}_rank{}_SEED{}.npz".format(n,m,horizon,FACTOR,SEED)
        if os.path.exists(PATH + cov_file):
            tmp = np.load(PATH + cov_file)
            model.pop_task_covar_module.covar_factor.data = torch.tensor(tmp["pop_factor"])

    # select parameters to learn
    if model_type=="both":
        # model.task_weights_module.weights = 0.5*torch.ones((n))
        # model.task_weights_module.raw_weights.requires_grad = True
        final_params = list(set(model.parameters()) - \
                    {model.fixed_module.raw_lengthscale}) + \
            list(likelihood.parameters())
    else:
        # model.task_weights_module.raw_weights.requires_grad = False
        if model_type=="pop":
            # model.task_weights_module.weights = torch.ones((n))
            final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale,\
                        # model.task_weights_module.raw_weights,\
                        }) + \
                       list(likelihood.parameters())
        elif model_type=="ind":
            # model.task_weights_module.weights = torch.zeros((n))
            model.pop_task_covar_module.raw_var.requires_grad = False
            model.pop_task_covar_module.covar_factor.requires_grad = False
            final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale,\
                        # model.task_weights_module.raw_weights,\
                        model.pop_task_covar_module.raw_var,
                        model.pop_task_covar_module.covar_factor}) + \
                        list(likelihood.parameters())

    optimizer = torch.optim.Adam(final_params, lr=0.1)
    
    num_params = 0
    for p in final_params:
        if p.requires_grad:
            num_param = np.prod(p.size())
            if num_param<num_inducing:
                num_params += num_param
    print("num of model parameters: {}".format(num_params))

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
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    # prediction
    model.eval()
    likelihood.eval()

    train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader)
    test_acc, test_ll = evaluate_gpr(model, likelihood, test_loader)

    loading_file = "loadings_n{}_m{}_t{}_rank{}_SEED{}.npz".format(n,m,horizon,RANK,SEED)
    PATH = "./data/synthetic/"
    dgp_loadings = np.load(PATH + loading_file)
    results = {}
    print("in-sample evaluatiion...")
    print("train acc: {}".format(train_acc))
    print("train ll: {}".format(train_ll))
    print("out-of-sample evaluatiion...")
    print("test acc: {}".format(test_acc))
    print("test ll: {}".format(test_ll))

    results["train_acc"] = train_acc
    results["train_ll"] = train_ll
    results["test_acc"] = test_acc
    results["test_ll"] = test_ll
    results["log_lik"] = log_lik
    results["BIC"] = num_params*np.log(train_x.size(0)) - 2*log_lik 

    if model_type!="ind":
        task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        results["pop_covariance"] = task_kernel
        results["pop_factor"] = model.pop_task_covar_module.covar_factor.data.detach().numpy()
        dgp_pop_loadings = dgp_loadings["pop_loadings"]
        dgp_covariance = dgp_pop_loadings.T @ dgp_pop_loadings
        pop_dist = correlation_matrix_distance(dgp_covariance, results["pop_covariance"])
        print("pop dist: {}".format(pop_dist))

    unit_covariance = np.zeros((n,m,m))
    # weights = model.task_weights_module.weights.detach().numpy()
    # results["weights"] = weights
    dgp_unit_loadings = dgp_loadings["unit_loadings"]
    dgp_pop_loadings = dgp_loadings["pop_loadings"]
    for i in range(n):
        task_kernel = np.zeros((m,m))
        if model_type!="ind":
            task_kernel += model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        if model_type!="pop":
            task_kernel += model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy()
        unit_covariance[i] = task_kernel
        results["unit_{}_covariance".format(i)] = task_kernel
        dgp_covariance = dgp_pop_loadings.T @ dgp_pop_loadings + dgp_unit_loadings[i].T @ dgp_unit_loadings[i]
        unit_dist = correlation_matrix_distance(dgp_covariance, unit_covariance[i])
        # if model_type!="pop":
        #     print("unit {} dist: {}".format(i, correlation_matrix_distance(\
        #         dgp_pop_loadings.T @ dgp_pop_loadings,
        #         model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy())))
        print("unit {} dist: {}".format(i, unit_dist))
        # plot_task_kernel(dgp_covariance, np.arange(m), "./data/synthetic/dgp_{}.pdf".format(i), SORT=False)
        # loading_scales, unit_loadings = np.linalg.eig(task_kernel)
        # loading_idx = np.argpartition(loading_scales, -RANK)[-RANK:]
        # loading_scales[loading_idx] * unit_loadings[:,loading_idx]
        # plot_task_kernel(unit_covariance[i], np.arange(m), "./data/synthetic/est_{}.pdf".format(i), SORT=False)

    PATH = "./results/synthetic/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    cov_file = "cov_{}_n{}_m{}_t{}_rank{}_SEED{}.npz".format(model_type, n,m,horizon,FACTOR,SEED)
    np.savez(PATH+cov_file, **results)

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=5)
    # pca.fit(results["pop_covariance"])
    # print(pca.explained_variance_ratio_)
    # vecs = pca.components_
    
    # import matplotlib.pyplot as plt
    # for i in range(5):
    #     plt.plot(vecs[0,(i*4):(i*4+4)],vecs[1,(i*4):(i*4+4)], "x")
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])
    # plt.savefig("./results/pop/pop_pca.pdf")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-n num_unit -m num_item -t num_period -s seed -r rank -f factor')
    parser.add_argument('-n','--num_unit', help='number of units', required=False)
    parser.add_argument('-m','--num_item', help='number of items', required=False)
    parser.add_argument('-t','--num_period', help='number of periods', required=False)
    parser.add_argument('-s','--seed', help='random seed', required=False)
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-r','--rank', help='rank of item correlation matrix', required=False)
    parser.add_argument('-f','--factor', help='number of coregionalization factors', required=False)
    args = vars(parser.parse_args())
    main(args)