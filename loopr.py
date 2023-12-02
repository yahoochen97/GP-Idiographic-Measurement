# loading packages
import numpy as np
import torch
import os
import pandas as pd
import gpytorch
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

def main():
    # load data
    load_batch_size = 512
    num_inducing = 1000
    num_epochs = 5
    model_type="pop"
    print("loading data...")
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])

    n = data.shape[0]
    m = data.shape[1]
    Items = data.columns
    horizon = 1
    Q = 5
    C = 5

    # x: [i,j,h]
    train_x = torch.zeros((n*m*horizon,3))
    train_y = torch.zeros((n*m*horizon,))

    iter = 0
    for i in range(n):
        for j in range(m):
            train_x[iter, 0] = i
            train_x[iter, 1] = j
            train_x[iter, 2] = horizon
            train_y[iter] = data.iloc[i][j]
            iter += 1

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)

    # initialize likelihood and model
    
    inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    likelihood = OrdinalLikelihood(thresholds=torch.tensor([-5.,-2.,-1.,1.,2.,5.]))
    model = OrdinalLMC(inducing_points,n,m,C,rank=Q, model_type=model_type)

    model.train()
    likelihood.train()

    for i in range(n):
        model.t_covar_module[i].lengthscale = 1
    model.fixed_module.raw_lengthscale.requires_grad = False
    final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale}) + \
                       list(likelihood.parameters())

    # Our loss object. We're using the VariationalELBO
    optimizer = torch.optim.Adam(final_params, lr=0.1)
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
            log_lik += loss.item()*y_batch.shape[0]
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    
    print("in-sample evaluatiion...")
    model.eval()
    likelihood.eval()
    train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader)
    print("train acc: {}".format(train_acc))
    print("train ll: {}".format(train_ll))

    directory = "./results/" +  model_type
    if not os.path.exists(directory):
        os.makedirs(directory)

    task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
    file_name = "./results/" + model_type + "/loopr_rank_{}.pdf".format(Q)
    plot_task_kernel(task_kernel, Items, file_name, SORT=False)

    results = {}
    results["pop_covariance"] = task_kernel
    PATH = "./results/"
    cov_file = "loopr_rank{}.npz".format(Q)
    np.savez(PATH+cov_file, **results)
   
if __name__=="__main__":
    main()