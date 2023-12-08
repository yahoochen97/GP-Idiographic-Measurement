# loading packages
import numpy as np
import torch
import os
import pandas as pd
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
    num_inducing = 60
    num_epochs = 10
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
            train_x[iter, 0] = 0
            train_x[iter, 1] = j
            train_x[iter, 2] = 0
            train_y[iter] = data.iloc[i][j]
            iter += 1

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)

    # initialize likelihood and model
    
    inducing_points = train_x[:num_inducing,:]
    likelihood = OrdinalLikelihood(thresholds=torch.tensor([-5.,-2.,-1.,1.,2.,5.]))
    model = OrdinalLMC(inducing_points,n=1,m=m,C=C,horizon=horizon,pop_rank=Q, model_type=model_type)

    model.train()
    likelihood.train()

    # initialize covariance of pop factors
    cov = torch.tensor(data.corr().to_numpy())
    _, _, V = torch.pca_lowrank(cov, q = Q)
    model.pop_task_covar_module.covar_factor.data = 4*torch.matmul(cov, V[:,:Q])

    # fix time length scale
    for i in range(1):
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
            log_lik += -loss.item()*y_batch.shape[0]
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    print("in-sample evaluatiion...")
    model.eval()
    likelihood.eval()
    train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader)
    print("train acc: {}".format(train_acc))
    print("train ll: {}".format(train_ll))

    directory = "./results/loopr/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
    results = {}
    results["pop_covariance"] = task_kernel
    results["pop_factor"] = model.pop_task_covar_module.covar_factor.data.detach().numpy()
    cov_file = "loopr_pop.npz".format(Q)
    np.savez(directory+cov_file, **results)

    # task_kernel = np.clip(task_kernel,-4,4) / 4
    file_name = directory + "/loopr_pop.pdf"
    item_order = sorted(range(len(Items)), key=lambda k: Items[k])
    plot_task_kernel(task_kernel[item_order,:][:,item_order], Items[item_order], file_name, SORT=False)

def cor_pca():
    data = np.load("./results/loopr/loopr_pop.npz")
    print(list(data.keys()))
    cov = data["pop_covariance"]
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    Items = data.columns
    item_order = sorted(range(len(Items)), key=lambda k: Items[k])
    cov = cov[item_order,:][:,item_order]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    pca.fit(cov)
    print(pca.explained_variance_ratio_)
    vecs = pca.components_
    pca.fit(vecs)
    
    import matplotlib.pyplot as plt
    MARKERS = ["+", "x", "*", "o", "D"]
    for i in range(5):
        plt.scatter(vecs[0,(i*4):(i*4+4)],vecs[1,(i*4):(i*4+4)], marker=MARKERS[i], label="factor_{}".format(i))
    plt.legend(loc=0)
    plt.savefig("./results/loopr/loopr_pca.pdf")
    plt.close()

def cor_factor():
    data = np.load("./results/loopr/loopr_pop.npz")
    print(list(data.keys()))
    cov = data["pop_factor"]
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    # Items = data.columns
    # item_order = sorted(range(len(Items)), key=lambda k: Items[k])
    # cov = cov[item_order,:]
    
    import matplotlib.pyplot as plt
    plt.plot(cov[:,0],cov[:,1], "x")
    plt.xlabel("factor 1")
    plt.ylabel("factor 2")
    plt.savefig("./results/loopr/loopr_factor12.pdf")
    plt.close()
    plt.plot(cov[:,2],cov[:,3], "x")
    plt.xlabel("factor 3")
    plt.ylabel("factor 4")
    plt.savefig("./results/loopr/loopr_factor34.pdf")
    plt.close()
    plt.plot(cov[:,0],cov[:,4], "x")
    plt.xlabel("factor 1")
    plt.ylabel("factor 5")
    plt.savefig("./results/loopr/loopr_factor15.pdf")
    plt.close()

if __name__=="__main__":
    # main()
    # cor_factor()
    cor_pca()