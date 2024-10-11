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

# different seed 
SEED=3407
torch.manual_seed(SEED)
np.random.seed(SEED)

import warnings
warnings.filterwarnings("ignore")

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader
from utilities.util import OrdinalLMC, OrdinalLikelihood
from utilities.util import correlation_matrix_distance, plot_task_kernel
from utilities.util import plot_agg_task_kernel, evaluate_gpr

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

    # read new data
    data = pd.read_csv("./data/GP_Data.csv", index_col=[0])
    # data = data[data.PID<10]

    data.columns = [x.replace(" ", "") for x in data.columns]
    ESM_items = [x.replace(" ", "") for x in codebook.iloc[:,0].to_list() if x.replace(" ", "") in Items_loopr]
    reverse_code = [reverse_code[i] for i in range(codebook.shape[0]) if codebook.iloc[i,0].replace(" ", "") in Items_loopr]
    reverse_code = np.array(reverse_code).reshape(-1,1)
    time_diff = (pd.to_datetime(data.RecordedDate, format='%Y-%m-%d %H:%M:%S')-\
                 pd.to_datetime(data.RecordedDate.iloc[0])).dt
    data["day"] = time_diff.days
    data["day"] += time_diff.seconds/60/60/25

    # cols = [item_mapping[x] for x in ESM_items] + ["PID", "day"]
    # data = data[cols]
    # data.to_csv("./data/GP_ESM.csv", index=False)

    PIDs = data.PID.unique()
    n = PIDs.shape[0]
    PID_mapping = dict(zip(PIDs, range(n)))
    m = len(ESM_items)
    horizon = data.day.max()

    # transform to row data frame
    train_x = torch.zeros((n*m*data.n.max(),3))
    train_y = torch.zeros((n*m*data.n.max(),))

    ITER = 0
    for iter in range(data.shape[0]):
        for j in range(m):
            train_x[ITER, 0] = PID_mapping[data.PID.iloc[iter]]
            train_x[ITER, 1] = j
            train_x[ITER, 2] = data.day.iloc[iter]
            train_y[ITER] = data[item_mapping[ESM_items[j]]].iloc[iter]
            if reverse_code[j,0]==1:
                train_y[ITER] = 6 - train_y[ITER]
            ITER += 1

    train_x = train_x[~train_y.isnan()]
    train_y = train_y[~train_y.isnan()]
    train_x = train_x[train_y!=0]
    train_y = train_y[train_y!=0]

    print(train_x.shape)

    # build data loader
    C = 5
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)

    # initialize likelihood and model
    inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
    if model_type=="Gaussian":
        likelihood = GaussianLikelihood()
        model_type = "both"
    else:
        likelihood = OrdinalLikelihood(thresholds=torch.tensor([-20.,-2.5,-1.,1.,2.5,20.]))
    pop_rank = 5
    unit_rank = FACTOR
    if model_type=="pop":
        pop_rank = FACTOR
    model = OrdinalLMC(inducing_points,n=n,m=m,C=C,horizon=horizon,\
                    pop_rank=pop_rank, unit_rank=unit_rank, model_type=model_type)

    model.train()
    likelihood.train()

    # initialize covariance of pop factors
    pop_prior = np.load("./results/loopr{}/loopr_pop_f{}_e10.npz".format(SEED, pop_rank))
    loopr_idx = [Items_loopr.index(x) for x in ESM_items]
    model.pop_task_covar_module.covar_factor.data = torch.tensor(pop_prior["pop_factor"][loopr_idx])
    model.pop_task_covar_module.covar_factor.requires_grad = False
    if model_type=="ind":
        for i in range(n):
            model.unit_task_covar_module[i].covar_factor.data = torch.tensor(pop_prior["pop_factor"][loopr_idx])
    # if model_type=="both":
    #     for i in range(n):
    #         model.unit_task_covar_module[i].covar_factor.data *= 0 
    
    # select hyperparameters to learn
    for i in range(n):
        model.t_covar_module[i].lengthscale = 7 # data.day.max() // 3
    model.fixed_module.raw_lengthscale.requires_grad = False

    final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale, model.pop_task_covar_module.covar_factor}) + \
                    list(likelihood.parameters())

    num_params = 0
    for p in final_params:
        if p.requires_grad:
            num_param = np.prod(p.size())
            if num_param<num_inducing:
                num_params += num_param
    print("num of model parameters: {}".format(num_params))

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
            if j % 50:
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
    results["BIC"] = num_params*np.log(train_x.size(0)) - 2*log_lik 

    task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
    results["pop_covariance"] = task_kernel
    results["pop_factor"] = model.pop_task_covar_module.covar_factor.data.detach().numpy()

    unit_covariance = np.zeros((n,m,m))
    for i in range(n):
        task_kernel = np.zeros((m,m))
        if model_type!="ind":
            task_kernel += model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        if model_type!="pop":
            task_kernel += model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy()
        unit_covariance[i] = task_kernel
        results["unit_{}_covariance".format(i)] = task_kernel
        results["unit_{}_factor".format(i)] = model.unit_task_covar_module[i].covar_factor.detach().numpy()

    PATH = "./results/GP_ESM_2/"
    PATH = "./results/GP_ESM_{}/".format(SEED)
    if isinstance(likelihood, GaussianLikelihood):
        model_type = "Gaussian"
        PATH = "./results/GP_ESM_2/baselines/"

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    np.savez(PATH+"{}_f{}.npz".format(model_type, FACTOR), **results)

def plot_unit_cor_matrix():
    PATH = "./results/GP_ESM/"
    results = np.load(PATH+"both_f1_Feb.npz") # Feb

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

    all_cov = np.zeros((n,len(ESM_items),len(ESM_items)))
    # item_order = sorted(range(len(ESM_items)), key=lambda k: ESM_items[k])
    # plot populational kernel
    pop_task_kernel = results["pop_covariance"]
    pop_task_kernel = pop_task_kernel # * reverse_mask
    plot_task_kernel(pop_task_kernel, np.array(ESM_items), "./results/GP_ESM/both_5.pdf", SORT=False)
    # plot individual kernel
    for i in range(n):
        ind_task_kernel = results["unit_{}_covariance".format(i)]
        ind_task_kernel = ind_task_kernel # * reverse_mask
        all_cov[i] = ind_task_kernel
        plot_task_kernel(ind_task_kernel, \
                         np.array(ESM_items), \
                        "./results/GP_ESM/both_unit_cov/both_unit_{}_5.pdf".format(i), SORT=False)
        
    #     plot_agg_task_kernel(ind_task_kernel, pop_task_kernel, \
    #                      np.array(ESM_items), \
    #                     "./results/GP_ESM/both_ind_cov/both_ind_{}_5.pdf".format(i), SORT=False)

    return
    # iterate over all pairs of items
    for p in range(15):
        for q in range(p+1,15):
            item_cov = all_cov[:,(3*p):(3*p+3), (3*q):(3*q+3)]
            # 3 by 3 figures of individual level covariances
            plt.close()
            fig, axs = plt.subplots(3,3)
            for i in range(3):
                for j in range(3):
                    axs[i,j].boxplot(item_cov[:,i,j])
                    axs[i,j].set_ylim([-1.5,1.5])
                    axs[i,j].set_xticks([])

            for i in range(3):
                axs[i,0].set_ylabel(ESM_items[q*3+i])
                axs[i,0].get_yaxis().set_label_coords(-0.2,0.5)
            for j in range(3):
                axs[2,j].set_xlabel(ESM_items[p*3+j])
                    
            parts1 = ESM_items[p*3].split(".")
            parts1 = parts1[0]
            parts2 = ESM_items[q*3].split(".")
            parts2 = parts2[0]
            plt.savefig("./results/GP_ESM/both_pair_item/" + parts1 + "_" + parts2 + ".pdf", bbox_inches='tight')


def cluster_analysis():
    PATH = "./results/GP_ESM/"
    results = np.load(PATH+"both_f1_Feb.npz") # Feb

    data = pd.read_csv("./data/loopr_data.csv", index_col=[0])
    Items_loopr = data.columns.to_list()

    # rename volat to violat
    for i in range(1,5):
        Items_loopr[Items_loopr.index("Volat.{}".format(i))] = "Violat.{}".format(i)

    # generate item map from original to current using ESM codebook
    codebook = pd.read_excel("./data/ESM_Codebook.xlsx")
    # reverse_code = codebook.iloc[:,2].to_list()
    # reverse_code = [reverse_code[i] for i in range(codebook.shape[0]) if codebook.iloc[i,0].replace(" ", "") in Items_loopr]
    # reverse_code = np.array(reverse_code)#.reshape(-1,1)
    # reverse_mask = np.ones((reverse_code.shape[0],reverse_code.shape[0]))
    # reverse_mask[reverse_code==1,:] *= -1
    # reverse_mask[:,reverse_code==1] *= -1
    ESM_items = [x.replace(" ", "") for x in codebook.iloc[:,0].to_list() if x.replace(" ", "") in Items_loopr]
    data = pd.read_csv("./data/GP_ESM_cleaned.csv")
    n = data.PID.unique().shape[0]

    # plot populational kernel
    pop_task_kernel = results["pop_covariance"]
    pop_task_kernel = pop_task_kernel # * reverse_mask
    directory = "./results/GP_ESM/centroids/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_task_kernel(pop_task_kernel, \
            np.array(ESM_items), \
            directory + "both_pop_5.pdf", SORT=False)
    
    # print performance evaluation
    print(results["train_acc"])
    print(results["train_ll"])
    print(results["log_lik"])

    # plot individual kernel
    unit_cov_evs = np.zeros((n,5))
    discrepancy_pop = np.zeros((n,))
    all_cov = np.zeros((n,len(ESM_items),len(ESM_items)))
    for i in range(n):
        ind_task_kernel = results["unit_{}_covariance".format(i)]
        ind_task_kernel = ind_task_kernel # * reverse_mask
        all_cov[i] = ind_task_kernel
        eigv = np.linalg.eigvals(ind_task_kernel)
        eigv = np.sort(eigv)[::-1]
        unit_cov_evs[i] = np.around(eigv[0:5]/np.sum(eigv[0:5]), decimals=3)
        discrepancy_pop[i] = correlation_matrix_distance(pop_task_kernel, ind_task_kernel)

    # k mean clustering
    from utilities.util import matrix_cluster, matrix_kmeans
    matrix_cluster(all_cov, max_K=10)
    K = 5
    centroids, assignments, dists = matrix_kmeans(all_cov, K=K)
    # plot centroids
    for k in range(K):
        # plot_task_kernel(centroids[k], \
        #     np.array(ESM_items), \
        #     directory + "centroid_{}_5.pdf".format(k), SORT=False)
        plot_agg_task_kernel(centroids[k], pop_task_kernel, directory + "residual_{}.pdf".format(k))
    centroids_dist =  np.zeros((K,K))       
    for k in range(K):
        for k_ in range(K):
            centroids_dist[k, k_] = correlation_matrix_distance(centroids[k], centroids[k_])
    centroids_dist = np.around(centroids_dist, decimals=2)
    print(centroids_dist)

    discrepancy_pop = pd.DataFrame({"dist": discrepancy_pop,\
                                    "unit": np.arange(1,n+1),\
                                    "cluster": assignments})
    for i in range(5):
        discrepancy_pop["eig_{}".format(i+1)] = unit_cov_evs[:,i]

    discrepancy_pop.sort_values(by=["cluster", "dist"], inplace=True)
    COLORS = ["#d7191c", '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
    COLORS = ['#1b9e77',
            '#d95f02',
            '#7570b3',
            '#e7298a',
            '#66a61e']
    COLORS = ['#d73027',
            '#f46d43',
            '#fdae61',
            '#fee090',
            '#e0f3f8',
            '#abd9e9',
            '#74add1',
            '#4575b4']
    
    # import mantel
    # for k in range(K):
    #     mant = mantel.test(centroids[0], centroids[k], perms=10000, method='pearson', tail='upper')
    #     print(mant.summary())
    #     print("cluster {}: ".format(k+1))
    #     print(np.arange(1,n+1)[assignments==k])
    #     print(len(np.arange(1,n+1)[assignments==k]))
 
    # Add empty bars to the end of each group
    PAD = 1
    ANGLES_N = n + PAD * K
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False) 
    # Obtain size of each group
    GROUPS_SIZE = [len(i[1]) for i in discrepancy_pop.groupby("cluster")]
    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD
    COLORS_ = [COLORS[i] for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    # plot pie plot of clustering by discrepancy to the population model 
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_direction(-1)
    ax.set_theta_offset( np.pi)
    ax.set_xticks([])
    ax.set_ylim(0, np.max(discrepancy_pop.dist)*1.1)
    ax.bar(ANGLES[IDXS], discrepancy_pop.dist, alpha=0.7, width=6/n,\
           color=COLORS_, edgecolor="white", linewidth=1, zorder=5)
    # ax.set_ylabel('dist to pop kernel', fontsize=20)  
    plt.savefig("./results/GP_ESM/discrepancy_bar.pdf", bbox_inches='tight')

    # plot pie plot of eigen value
    discrepancy_pop.sort_values(by=["cluster", "unit"], inplace=True)
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "polar"}) 
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    bottom = np.zeros(n)
    for i in range(5):
        ax.bar(ANGLES[IDXS], discrepancy_pop["eig_{}".format(i+1)], alpha=0.9-0.2*i, width=6/n,\
            bottom=bottom, color=COLORS_, edgecolor="white", linewidth=1)
        bottom += discrepancy_pop["eig_{}".format(i+1)]
    for i in range(n):
        ax.annotate(discrepancy_pop.unit.values[i], (ANGLES[IDXS][i], 1.02), ha='center')   
    # ax.set_ylabel('relative proportions of eigen values', fontsize=20)  
    plt.savefig("./results/GP_ESM/eigvs.pdf", bbox_inches='tight')
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-k model_type -f factor')
    parser.add_argument('-k','--model_type', help='type of model', required=False)
    parser.add_argument('-f','--factor', help='number of coregionalization factors', required=False)
    args = vars(parser.parse_args())
    main(args)
    # plot_unit_cor_matrix()
    # cluster_analysis()
