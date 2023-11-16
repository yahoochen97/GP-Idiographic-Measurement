import numpy as np
import matplotlib.pylab as plt
import torch
import os
import argparse
import pandas as pd 
from scipy.stats import norm
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal

torch.set_default_dtype(torch.float64)

def main(args):
    SEED = int(args["seed"])
    n = int(args["num_unit"])
    m = int(args["num_item"])
    horizon = int(args["num_period"])
    RANK = int(args["rank"])

    # set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)    

    # generate latent trait vector
    x = torch.zeros((n, horizon, RANK))
    for i in range(n):
        x_cov = RBFKernel()
        x_cov.lengthscale = horizon // 3 if horizon > 1 else 1
        x_cov.lengthscale = (1+np.random.choice(3, 1)[0]) * x_cov.lengthscale
        x_cov = x_cov(torch.arange(horizon)).evaluate()/4 + 1e-6*torch.eye(horizon)
        for r in range(RANK):
            x[i,:,r] = torch.tensor(np.random.choice(3, 1)-1) + \
                MultivariateNormal(torch.zeros(horizon), x_cov).sample()
    x = x.numpy()
            
    # generate task correlation matrix
    pop_loadings = np.zeros((RANK,m))
    unit_loadings = np.zeros((n,RANK,m))
    for r in range(RANK):
        mask = np.zeros(RANK,).astype(bool)
        mask[r] = 1
        m_ = m//RANK
        high_loadings = np.array([-3 for _ in range(m_//2)] + [3 for _ in range((m_+1)//2)])
        np.random.shuffle(high_loadings)
        pop_loadings[mask,(r*m_):(r*m_+m_)] = high_loadings
        pop_loadings[~mask,(r*m_):(r*m_+m_)] = np.random.uniform(low=-1,high=1,size=((RANK-1),m//RANK))
        for i in range(n):
            unit_loadings[i,:,(r*m_):(r*m_+m_)] = np.random.uniform(low=-1,high=1, size=(RANK,m//RANK))
    
    drop_mask = np.zeros((n*RANK*m,))
    drop_mask[np.random.choice(n*RANK*m, int(n*RANK*m*0.5),replace=False)] = 1
    unit_loadings[drop_mask.reshape((n,RANK,m))==1] = 0
    
    # build dataset
    results = np.zeros((n*m*horizon, 1+1+1+1))
    iter = 0
    for i in range(n):
        for j in range(m):
            for h in range(horizon):
                results[iter,0] = i
                results[iter,1] = j
                results[iter,2] = h
                f = x[i,h].T @ (pop_loadings[:,j] + unit_loadings[i,:,j])
                results[iter,3] = f
                iter += 1

    results[:,3] = (results[:,3] - results[:,3].min()) / (results[:,3].max()-results[:,3].min()) * 6
    for iter in range(results.shape[0]):
        f = results[iter,3]
        if f <= 1:
            results[iter,3] = 1
        elif f>=5:
            results[iter,3] = 5
        else:
            results[iter,3] = np.rint(f)
            # np.random.choice([np.floor(f),np.ceil(f)],\
            #                 p=[f-np.floor(f),np.ceil(f)-f])           

    plt.hist(results[:,3])
    plt.show()
    results = pd.DataFrame(results)
    results.columns = ["unit","item","time","y"]

    print("splitting data for training/testing...")
    train_ratio = 0.8
    train_mask = np.zeros((results.shape[0],))
    train_mask[np.random.choice(results.shape[0], int(results.shape[0]*train_ratio),replace=False)] = 1
    results["train"] = train_mask

    PATH = "./data/synthetic/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    data_file = "data_n{}_m{}_t{}_rank{}_SEED{}.csv".format(n,m,horizon,RANK,SEED)
    results.to_csv(PATH + data_file, index=False)

    results = {}
    results["pop_loadings"] = pop_loadings
    results["unit_loadings"] = unit_loadings
    loading_file = "loadings_n{}_m{}_t{}_rank{}_SEED{}.npz".format(n,m,horizon,RANK,SEED)
    np.savez(PATH+loading_file, **results)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='-n num_unit -m num_item -t num_period -s seed')
    parser.add_argument('-n','--num_unit', help='number of units', required=False)
    parser.add_argument('-m','--num_item', help='number of items', required=False)
    parser.add_argument('-t','--num_period', help='number of periods', required=False)
    parser.add_argument('-s','--seed', help='random seed', required=False)
    parser.add_argument('-r','--rank', help='rank of item correlation matrix', required=False)
    args = vars(parser.parse_args())
    main(args)