import numpy as np
import matplotlib.pylab as plt
import torch
import os
import gpytorch
from gpytorch.mlls import VariationalELBO

from scipy.stats import norm
torch.manual_seed(8927)
np.random.seed(8927)
torch.set_default_dtype(torch.float64)

from gpytorch.models import ApproximateGP, ExactGP
from matplotlib import pyplot as plt
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.means import ZeroMean, LinearMean, MultitaskMean
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import LinearKernel, RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from torch.nn import ModuleList

from utilities.util import VariationalDynamicLatentVariable, MAPDynamicLatentVariable
from utilities.util import MaskMean, ConstantVectorMean, LinearMeanWithPrior, OrdinalLikelihood

class GPIRT(ApproximateGP):
    def __init__(self, n, m, horizon, latent_dim, n_inducing, X_init, est_method="MAP"):
        self.n = n # infer n*horizon latent variables 
        self.horizon = horizon # time series of length horizon
        self.m = m
        self.batch_shape = torch.Size([m]) # infer m weights
        self.latent_dim = latent_dim

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        # first data_dim for x and second data_dim for w
        # self.inducing_inputs = torch.randn(m, n_inducing, latent_dim)
        self.inducing_inputs = torch.linspace(-3,3,n_inducing).t()
        # self.inducing_inputs = torch.linspace(-3,3,n_inducing).t().unsqueeze(0).repeat(2, 1)
        # self.inducing_inputs = self.inducing_inputs.unsqueeze(2).repeat(1,1,horizon*latent_dim)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, \
                                  learn_inducing_locations=False)
        super().__init__(q_f)

        # Define prior for X
        X_prior_mean = torch.zeros(horizon*latent_dim,)  # shape: 1 x Q
        X_prior_cov = RBFKernel()
        X_prior_cov.lengthscale = horizon // 3 if horizon > 1 else 1
        X_prior_cov = X_prior_cov(torch.arange(horizon)).evaluate() \
              + 4*torch.ones((horizon,horizon)) + 1e-6*torch.eye(horizon)
        # TODO: K_dim subject to change
        X_prior_cov = torch.kron(torch.eye(latent_dim),X_prior_cov)
        prior_x = MultivariateNormalPrior(X_prior_mean, covariance_matrix=X_prior_cov)

        if est_method=="VI":
            # variational inference for LatentVariable (c)
            X = VariationalDynamicLatentVariable(n, m, horizon, latent_dim, X_init, prior_x)
        elif est_method=="MAP":
        # For (a) or (b) change to below:
        # X = PointDynamicLatentVariable(n, horizon, latent_dim, X_init)
        # prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            X = MAPDynamicLatentVariable(n, horizon, latent_dim, X_init, prior_x)
        self.X = X

        # Kernel (acting on latent dimensions)
        # self.mean_module = ModuleList([LinearMeanWithPrior(input_size=latent_dim, \
        #                 weights_prior=NormalPrior(0.,2.)) for _ in range(m)])
        self.mean_module = ZeroMean()
        self.covar_module = LinearKernel(ard_num_dims=latent_dim)
    
    def sample_latent_variable(self):
        sample = self.X()
        return sample

    def forward(self, X):
        # X: n*horizon + num_inducing
        # m = len(self.mean_weights)
        m = self.m
        mean_x = torch.zeros((m, X.shape[0]))
        for j in range(m):
            # X_ = X[j].reshape((-1, self.horizon, self.latent_dim)) # n x T x Q
            mean_x[j] = self.mean_module(X)
        # mean_x = torch.permute(mean_x, (2,0,1)).reshape(X.shape[1],m*self.horizon).T

        # X_ = X.reshape((m,-1,self.horizon,self.latent_dim))
        # X_ = torch.permute(X_, (0,2,1,3))
        # X_ = X_.reshape((m*self.horizon,-1, self.latent_dim))
        covar_x = self.covar_module(X).unsqueeze(0).repeat(m, 1, 1)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size, iter):
        valid_indices = np.arange(self.n*self.horizon)
        iter_ = iter % (self.n*self.horizon//batch_size)
        batch_indices = valid_indices[(batch_size*iter_):(batch_size*iter_+batch_size)]
        return np.sort(batch_indices)

class NewGPIRT(ApproximateGP):
    def __init__(self,  n, m, horizon, latent_dim):
        self.n = n
        self.m = m
        self.horizon = horizon
        self.latent_dim = latent_dim

        self.inducing_inputs = torch.arange(horizon).t().unsqueeze(0).repeat(n, 1)
        self.inducing_inputs = self.inducing_inputs.unsqueeze(2).repeat(1,1,latent_dim)

        # Variational Formulation
        q_u = CholeskyVariationalDistribution(horizon)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, \
                                  learn_inducing_locations=False)
        super().__init__(q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean()
        self.register_parameter("weights", parameter=torch.nn.Parameter(torch.randn(m, latent_dim)))
        self.covar_module = RBFKernel(active_dims=[2])

    def forward(self, X):
        mean_x = self.mean_module(X[:,[0,2]])
        mean_x *= self.weights[X[:,1].long()].reshape(-1,)

        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.train_x.shape[0])
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

def new_main():
    n = 20
    m = 10
    horizon = 3
    latent_dim = 1
    
    true_slope = np.concatenate((np.random.normal(2,0.5,size=(m//2,1)),
                    np.random.normal(-2,0.5,size=(m//2,1)))).reshape(m,latent_dim)
    # true_slope = np.repeat(true_slope, horizon, axis=1)
    x = torch.zeros((n, horizon))
    x_cov = RBFKernel()
    x_cov.lengthscale = horizon // 3 if horizon > 1 else 1
    x_cov = x_cov(torch.arange(horizon)).evaluate()
    for i in range(n):
        x[i] = (2*torch.bernoulli(torch.tensor(0.5))-1) + \
            MultivariateNormal(torch.zeros(horizon), x_cov+1e-6*torch.eye(horizon)).sample()
    
    # nx (m*horizon)
    train_x = np.zeros((n*m*horizon, 3))
    train_y = np.zeros((n*m*horizon,))
    counter = 0
    for i in range(n):
        for j in range(m):
            for h in range(horizon):
                train_x[counter,0] = i
                train_x[counter,1] = j
                train_x[counter,2] = h
                train_y[counter] = np.random.binomial(1, norm.cdf(true_slope[j] * x[i,h].numpy()))
                counter += 1
    print(train_y.shape)
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y).long()
    
    # Initialise X with randn
    X_init = torch.nn.Parameter(torch.randn((n,latent_dim*horizon)))

    # Model
    likelihood = BernoulliLikelihood()
    model = NewGPIRT(n, m, horizon, latent_dim)

    # Declaring the objective to be optimised along with optimiser
    mll = VariationalELBO(likelihood, model, num_data=n*horizon*m)

    # fix variance of prior on x to be 0.01.
    hypers = {
        'covar_module.lengthscale': torch.tensor(horizon / 3)
    }

    model.initialize(**hypers)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()}
    ], lr=0.05)

    loss_list = []
    # batch_size = n
    # for i in range(500):
    #     batch_index = model._get_batch_idx(batch_size)
    #     optimizer.zero_grad()
    #     sample_batch = train_x[batch_index]
    #     output_batch = model(sample_batch)
    #     loss = -mll(output_batch, train_y[batch_index]).sum()
    #     loss_list.append(loss.item())
    #     loss.backward(retain_graph=True)
    #     optimizer.step()

    for i in range(200):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        if i % 5 == 0:
            print('Iter %d/%d - Loss: %.3f ' % (
                i + 1, 200, loss.item()
            ))
        loss_list.append(loss.item())
        optimizer.step()

    plt.plot(loss_list, label='batch_size=50')
    plt.title('Neg. ELBO Loss', fontsize='small')
    plt.show()

    model.eval()
    likelihood.eval()
    
    X = model.mean_module.constantvector.detach().numpy()
    plt.scatter(x, X)
    plt.xlabel("true_x")
    plt.ylabel("est_x")
    plt.show()

    est_slopes = model.weights.data.detach().numpy()

    plt.scatter(true_slope, est_slopes)
    plt.xlabel("true_w")
    plt.ylabel("est_w")
    plt.show()
    print(est_slopes)


def main():
    n = 20
    m = 10
    horizon = 15
    latent_dim = 1
    
    true_slope = np.concatenate((np.random.normal(2,0.5,size=(m//2,1)),
                    np.random.normal(-2,0.5,size=(m//2,1)))).reshape(m,latent_dim)

    x = torch.zeros((n, horizon))
    x_cov = RBFKernel()
    x_cov.lengthscale = horizon // 3 if horizon > 1 else 1
    x_cov = x_cov(torch.arange(horizon)).evaluate()
    for i in range(n):
        x[i] = (2*torch.bernoulli(torch.tensor(0.5))-1) + \
            MultivariateNormal(torch.zeros(horizon), x_cov+1e-6*torch.eye(horizon)).sample()
    
    true_p = np.zeros((n, horizon, m))
    for i in range(n):
        for j in range(m):
            true_p[i,:,j] = norm.cdf(true_slope[j] * x[i].numpy())
    y = np.random.binomial(1, true_p)
    y = torch.tensor(y).reshape((n*horizon,m))
    print(y.shape)
    
    n_inducing = 30
    # Initialise X with PCA
    # y_ = y.reshape((n,horizon,m)).reshape((n,m*horizon))
    # _, _, V = torch.pca_lowrank(y_.double(), q = latent_dim*horizon)
    # X_init = torch.matmul(y_.double(), V[:,:latent_dim*horizon])
    # X_init = torch.nn.Parameter(X_init / X_init.abs().max())
    # Initialise X with randn
    X_init = torch.nn.Parameter(torch.zeros((n,latent_dim*horizon)))

    # X_init = torch.nn.Parameter(MultivariateNormal(torch.zeros(horizon),\
    #            x_cov+1e-6*torch.eye(horizon)).sample(sample_shape=torch.Size([n])))

    # Model
    model = GPIRT(n, m, horizon, latent_dim, n_inducing, X_init, est_method="MAP")
    # likelihood = DirichletClassificationLikelihood(targets=y.view(-1), learn_additional_noise=True)
    likelihood = BernoulliLikelihood()
    # likelihood = OrdinalLikelihood(thresholds=torch.tensor([-5.,0.,5.]))

    # Declaring the objective to be optimised along with optimiser
    # (see models/latent_variable.py for how the additional loss terms are accounted for)
    batch_size = n*horizon // 5
    mll = VariationalELBO(likelihood, model, num_data=n*horizon*m)

    # fix variance of prior on w to be 0.01.
    hypers = {
        'covar_module.variance': torch.tensor(0.1**2),
    }
    # for j in range(m):
    #     hypers[f'mean_module.{j}.weights'] = torch.tensor(true_slope[j])

    model.initialize(**hypers)
    model.covar_module.raw_variance.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.1, betas=(0.9, 0.95))

    # initial x/w with zero and MAP estimation
    loss_list = []
    n_steps = 500
    for iter in range(n_steps):
        batch_index = model._get_batch_idx(batch_size, iter)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        sample_batch = sample.reshape((-1,1))[batch_index]
        output_batch = model(sample_batch)
        tmp = likelihood(output_batch)
        # manually iterate m questions
        loss = 0
        for j in range(m):
            output_batch_j = MultivariateNormal(output_batch.loc[j], output_batch.covariance_matrix[j])
            loss += -mll(output_batch_j, y[batch_index,j].T).sum()
        # loss = -mll(output_batch, y[batch_index].T).sum()
        loss_list.append(loss.item())
        loss.backward(retain_graph=True)
        if iter % 50 == 0:
            print("Elbo loss: {} for {}/{}".format(loss, iter, n_steps))

        optimizer.step()

    plt.plot(loss_list, label='MAP loss')
    plt.title('Neg. ELBO Loss', fontsize='small')
    plt.show()

    est_X = model.X.X.detach().numpy()
    plt.scatter(x, est_X)
    plt.xlabel("true_x")
    plt.ylabel("est_x")
    plt.show()

    # est_slopes = []
    # for i in range(m):
    #     # est_slopes.append(model.mean_weights.data[i])
    #     est_slopes.append(model.mean_module[i].weights.data)

    # plt.scatter(true_slope, est_slopes)
    # plt.xlabel("true_w")
    # plt.ylabel("est_w")
    # plt.show()

    return

    # Variational Inference model
    model = GPIRT(n, m, horizon, latent_dim, n_inducing, \
                  X_init=torch.nn.Parameter(torch.tensor(MAP_X)), est_method="VI")
    likelihood = BernoulliLikelihood()

    # Declaring the objective to be optimised along with optimiser
    # (see models/latent_variable.py for how the additional loss terms are accounted for)
    batch_size = n*horizon // 5
    mll = VariationalELBO(likelihood, model, num_data=n*horizon*m)

    # fix variance of prior on w to be 0.01.
    hypers = {
        'covar_module.variance': torch.tensor(0.1**2),
    }\

    model.initialize(**hypers)
    model.covar_module.raw_variance.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.1, betas=(0.9, 0.95))

    # initial x/w with zero and MAP estimation
    loss_list = []
    n_steps = 5000
    for iter in range(n_steps):
        batch_index = model._get_batch_idx(batch_size, iter)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        sample_batch = sample.reshape((-1,1))[batch_index]
        output_batch = model(sample_batch)
        # manually iterate m questions
        loss = 0
        for j in range(m):
            output_batch_j = MultivariateNormal(output_batch.loc[j], output_batch.covariance_matrix[j])
            loss += -mll(output_batch_j, y[batch_index,j].T).sum()
        # loss = -mll(output_batch, y[batch_index].T).sum()
        loss_list.append(loss.item())
        loss.backward(retain_graph=True)
        if iter % 50 == 0:
            print("Elbo loss: {} for {}/{}".format(loss, iter, n_steps))

        optimizer.step()

    plt.plot(loss_list, label='MAP loss')
    plt.title('Neg. ELBO Loss', fontsize='small')
    plt.show()

    VI_X = model.X.q_mu.detach().numpy()
    plt.scatter(x, VI_X)
    plt.xlabel("true_x")
    plt.ylabel("est_x")
    plt.title("VI")
    plt.show()

    est_slopes = []
    for i in range(m):
        # est_slopes.append(model.mean_weights.data[i])
        est_slopes.append(model.mean_module[i].weights.data)

    plt.scatter(true_slope, est_slopes)
    plt.xlabel("true_w")
    plt.ylabel("est_w")
    plt.title("VI")
    plt.show()


if __name__ == "__main__":
    main()