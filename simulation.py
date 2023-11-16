# loading packages
import numpy as np
import torch
import os
import pandas as pd
import gpytorch
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
from utilities.util import correlation_matrix_distance, plot_task_kernel

def main(args):
    SEED = int(args["seed"])
    n = int(args["num_unit"])
    m = int(args["num_item"])
    horizon = int(args["num_period"])
    RANK = int(args["rank"])
    model_type = args["model_type"]
    load_batch_size = 256
    num_inducing = 1000
    num_epochs = 20

    # load data
    print("loading data...")
    PATH = "./data/synthetic/"
    data_file = "data_n{}_m{}_t{}_rank{}_SEED{}.csv".format(n,m,horizon,RANK,SEED)
    data = pd.read_csv(PATH + data_file)
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
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = OrdinalLMC(inducing_points,n,m,C,rank=RANK, model_type=model_type)

    model.train()
    likelihood.train()

    # select hyperparameters to learn
    for i in range(n):
        model.t_covar_module[i].lengthscale = horizon // 3 if horizon > 1 else 1
    model.fixed_module.raw_lengthscale.requires_grad = False
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

    # Our loss object. We're using the VariationalELBO
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # train GPR
    print("start training...")
    for i in range(num_epochs):
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))

    # define utilities to evaluate
    def evaluate_gpr(data_loader):
        means = torch.tensor([0])
        true_ys = torch.tensor([0])
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            for x_batch, y_batch in data_loader:
                test_dist = likelihood(model(x_batch))
                if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
                    probabilities = test_dist.loc
                else:
                    probabilities = test_dist.probs.argmax(axis=0) + 1
                means = torch.cat([means, probabilities])
                true_ys = torch.cat([true_ys, y_batch])
        means = means[1:]
        true_ys = true_ys[1:]

        acc = torch.sum(torch.abs(true_ys-means)<=0.5) / true_ys.shape[0]
        acc1 = torch.sum(torch.abs(true_ys-means)<=1) / true_ys.shape[0]
        return acc, acc1

    model.eval()
    likelihood.eval()

    train_acc, train_acc1 = evaluate_gpr(train_loader)
    test_acc, test_acc1 = evaluate_gpr(test_loader)

    loading_file = "loadings_n{}_m{}_t{}_rank{}_SEED{}.npz".format(n,m,horizon,RANK,SEED)
    PATH = "./data/synthetic/"
    dgp_loadings = np.load(PATH + loading_file)
    results = {}
    print("in-sample evaluatiion...")
    print("train acc: {}".format(train_acc))
    print("out-of-sample evaluatiion...")
    print("test acc: {}".format(test_acc))

    results["train_acc"] = train_acc
    results["train_acc1"] = train_acc1
    results["test_acc"] = test_acc
    results["test_acc1"] = test_acc1

    if model_type!="ind":
        task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        results["pop_covariance"] = task_kernel
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
        if model_type!="pop":
            print("unit {} dist: {}".format(i, correlation_matrix_distance(\
                dgp_pop_loadings.T @ dgp_pop_loadings,
                model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy())))
        print("unit {} weighted dist: {}".format(i, unit_dist))
        # plot_task_kernel(dgp_covariance, np.arange(m), "./data/synthetic/dgp_{}.pdf".format(i), SORT=False)
        # loading_scales, unit_loadings = np.linalg.eig(task_kernel)
        # loading_idx = np.argpartition(loading_scales, -RANK)[-RANK:]
        # loading_scales[loading_idx] * unit_loadings[:,loading_idx]
        # plot_task_kernel(unit_covariance[i], np.arange(m), "./data/synthetic/est_{}.pdf".format(i), SORT=False)

    PATH = "./results/synthetic/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    cov_file = "cov_{}_n{}_m{}_t{}_rank{}_SEED{}.npz".format(model_type, n,m,horizon,RANK,SEED)
    np.savez(PATH+cov_file, **results)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-n num_unit -m num_item -t num_period -s seed')
    parser.add_argument('-n','--num_unit', help='number of units', required=False)
    parser.add_argument('-m','--num_item', help='number of items', required=False)
    parser.add_argument('-t','--num_period', help='number of periods', required=False)
    parser.add_argument('-s','--seed', help='random seed', required=False)
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-r','--rank', help='rank of item correlation matrix', required=False)
    args = vars(parser.parse_args())
    main(args)