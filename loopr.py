# loading packages
import numpy as np
import torch
import os
import argparse
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
from gpytorch.likelihoods import GaussianLikelihood
from utilities.util import correlation_matrix_distance, plot_task_kernel, evaluate_gpr

def main(args):
    FACTOR = int(args["factor"])
    init_type = args["init_type"]
    model_type = args["model_type"]
    # load data
    load_batch_size = 512
    num_inducing = 60
    num_epochs = int(args["epoch"])
    print("loading data...")
    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])

    n = data.shape[0]
    m = data.shape[1]
    Items = data.columns
    horizon = 1
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
    # inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    if model_type=="Gaussian":
        likelihood = GaussianLikelihood()
        model_type = "pop"
    else:
        likelihood = OrdinalLikelihood(thresholds=torch.tensor([-20.,-2.,-1.,1.,2.,20.]))
    pop_rank = FACTOR
    model = OrdinalLMC(inducing_points,n=1,m=m,C=C,horizon=horizon,pop_rank=pop_rank, model_type=model_type)

    model.train()
    likelihood.train()

    # initialize covariance of pop factors
    if init_type=="PCA":
        cov = torch.tensor(data.corr().to_numpy())
        _, _, V = torch.pca_lowrank(cov, q = FACTOR)
        model.pop_task_covar_module.covar_factor.data = 4*torch.matmul(cov, V[:,:FACTOR])
    elif init_type=="gpcm":
        tmp = pd.read_csv("./results/loopr/gpcm_multi_{}.csv".format(FACTOR), index_col=[0]).to_numpy()
        model.pop_task_covar_module.covar_factor.data = 4*torch.tensor(tmp)
    elif init_type=="grm":
        tmp = pd.read_csv("./results/loopr/graded_multi_{}.csv".format(FACTOR), index_col=[0]).to_numpy()
        model.pop_task_covar_module.covar_factor.data = 4*torch.tensor(tmp)
    elif init_type=="srm":
        tmp = pd.read_csv("./results/loopr/sequential_multi_{}.csv".format(FACTOR), index_col=[0]).to_numpy()
        model.pop_task_covar_module.covar_factor.data = 4*torch.tensor(tmp)
    elif init_type=="sem":
        tmp = pd.read_csv("./results/loopr/SEM_{}.csv".format(FACTOR), index_col=[0]).to_numpy()
        model.pop_task_covar_module.covar_factor.data = 4*torch.tensor(tmp)

    # fix time length scale
    for i in range(1):
        model.t_covar_module[i].lengthscale = 1
    model.fixed_module.raw_lengthscale.requires_grad = False
    final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale}) + \
                       list(likelihood.parameters())

    # Our loss object. We're using the VariationalELBO
    optimizer = torch.optim.Adam(final_params, lr=0.01)
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    num_params = 5 + m*FACTOR + m + 1 # likelihood + multi task + noise
    print("num of model parameters: {}".format(num_params))

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
    train_acc, train_ll = evaluate_gpr(model, likelihood, train_loader, mll)
    print("train acc: {}".format(train_acc))
    print("train ll: {}".format(train_ll))

    if isinstance(likelihood, GaussianLikelihood):
        model_type = "Gaussian"
        init_type = "Gaussian"

    directory = "./results/loopr/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
    results = {}
    log_lik = train_ll * train_x.size(0)
    results["train_acc"] = train_acc
    results["train_ll"] = log_lik
    results["BIC"] = num_params*np.log(train_x.size(0)) - 2*log_lik 
    results["pop_covariance"] = task_kernel
    results["pop_factor"] = model.pop_task_covar_module.covar_factor.data.detach().numpy()
    cov_file = "loopr_pop_{}_f{}_e{}.npz".format(init_type, FACTOR, num_epochs)
    np.savez(directory+cov_file, **results)

    if init_type!="PCA" or model_type=="Gaussian":
        return

    # file_name = directory + "/loopr_pop_f{}_e{}.pdf".format(FACTOR, num_epochs)
    # item_order = ["Sociab.1", "Sociab.2", "Sociab.3", "Sociab.4",
    #               "Assert.1", "Assert.2", "Assert.3", "Assert.4",
    #               "Energy.1", "Energy.2", "Energy.3", "Energy.4",
    #               "Compass.1", "Compass.2", "Compass.3", "Compass.4",
    #               "Respect.1", "Respect.2", "Respect.3", "Respect.4",
    #               "Trust.1", "Trust.2", "Trust.3", "Trust.4",
    #               "Organiz.1", "Organiz.2", "Organiz.3", "Organiz.4",
    #               "Product.1", "Product.2", "Product.3", "Product.4",
    #               "Respons.1", "Respons.2", "Respons.3", "Respons.4",
    #               "Anxiety.1", "Anxiety.2", "Anxiety.3", "Anxiety.4",
    #               "Depres.1", "Depres.2", "Depres.3", "Depres.4",
    #               "Volat.1", "Volat.2", "Volat.3", "Volat.4",
    #               "Curious.1", "Curious.2", "Curious.3", "Curious.4",
    #               "Aesth.1", "Aesth.2", "Aesth.3", "Aesth.4",
    #               "Creativ.1","Creativ.2","Creativ.3","Creativ.4"]
    # item_order = [Items.to_list().index(item) for item in item_order]
    # plot_task_kernel(task_kernel[item_order,:][:,item_order], Items[item_order], file_name, SORT=False)

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

def model_comparison():
    PATH = "./results/loopr/"
    FACTORS = [1,2,3,4,5]
    train_lls = np.zeros((3,1,len(FACTORS)))
    for i in range(len(FACTORS)):
        results = np.load(PATH+"loopr_pop_sem_f{}_e10.npz".format(FACTORS[i]))
        train_ll = results["train_ll"] # * 207540
        train_lls[0,0,i] = train_ll
        train_lls[1,0,i] = results["train_acc"]
        train_lls[2,0,i] = (5+45*FACTORS[i]+45)*np.log(3459*60) - 2*train_ll*3459*60 # results["BIC"] 

        # results = np.load(PATH+"loopr_pop_f{}_e10.npz".format(FACTORS[i]))
        # train_ll = results["train_ll"] # * 207540
        # train_lls[0,1,i] = train_ll
        # train_lls[1,1,i] = results["train_acc"]
        # train_lls[2,1,i] = results["BIC"] 

    print("ll:")
    print(train_lls[0]+306516.59085925)
    print("acc:")
    print(train_lls[1])
    print("BIC:")
    print(train_lls[2])
    # import matplotlib.pylab as plt
    # plt.figure(figsize=(12, 10))
    # plt.plot(FACTORS, train_lls[0]/207540, label="PCA")
    # plt.plot(FACTORS, train_lls[1]/207540, label="ours")
    # plt.legend(loc=0,fontsize=20)
    # directory = "./results/loopr/"
    # file_name = directory + "/compare_pop_factors.pdf"
    # plt.savefig(file_name, bbox_inches='tight')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-k model_type -i init_type -f factor -e epoch')
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-i','--init_type', help='type of init', required=False)
    parser.add_argument('-e','--epoch', help='num of training epochs', required=False)
    parser.add_argument('-f','--factor', help='number of coregionalization factors', required=False)
    args = vars(parser.parse_args())
    main(args)
    # cor_factor()
    # cor_pca()
    # model_comparison()