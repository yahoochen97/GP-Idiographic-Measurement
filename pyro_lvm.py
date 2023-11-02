import numpy as np
import matplotlib.pylab as plt
import torch
import pyro
from pyro.distributions import Normal, Bernoulli, MultivariateNormal
from pyro.distributions.constraints import positive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from pyro.infer import MCMC, NUTS

from scipy.stats import norm
torch.manual_seed(8927)
np.random.seed(8927)

def main():
    pyro.clear_param_store()
    n = 20
    m = 10
    horizon = 6
    true_slope = np.concatenate((np.random.normal(2,0.5,size=(m//2,)),
                                 np.random.normal(-2,0.5,size=(m//2,)))).reshape(m,1)
    true_x = np.concatenate((np.linspace(-2, -1, n//2),
                        np.linspace(1, 2, n//2))).reshape(n,1)
    true_x = np.tile(true_x, (1, horizon))

    for i in range(n):
        tmp = np.arange(horizon)
        from sklearn.metrics.pairwise import rbf_kernel
        X = np.arange(horizon).reshape(-1,1)
        cov = rbf_kernel(X, gamma = 9/horizon/horizon) + 1e-6*np.eye(horizon)

        true_x[i] += np.random.multivariate_normal(np.zeros(horizon,),cov=cov)

    true_p = norm.cdf(np.dot(true_x.reshape((n,horizon,1)),np.transpose(true_slope)))
    y = np.random.binomial(1, true_p)
    y = torch.tensor(y).reshape((n*m*horizon,1))

    # initialize variational mean with pca
    latent_dim = 1

    def model(y):
        # sample w from the mv normal prior
        w = torch.zeros((m,1))
        x = torch.zeros((n,horizon))
        for i in range(m):
            w[i] = pyro.sample(f"w_{i}", Normal(0, 1))
        for i in range(n):
            x[i] = pyro.sample(f"x_{i}", MultivariateNormal(torch.zeros(horizon), torch.tensor(cov)))
        p = torch.sigmoid(torch.matmul(x.reshape((n,horizon,1)), w.t())).reshape((n*m*horizon,1))

        return pyro.sample("obs", Bernoulli(probs=p), obs=y)

    def guide(y):
        # register variational parameters
        w_mu = pyro.param("w_mu", torch.zeros(m,))
        w_std = pyro.param("w_std", 0.05*torch.ones(m,), constraint=positive)

        x_mu = pyro.param("x_mu", torch.zeros((n,horizon)))
        x_std = pyro.param("x_std", 0.05*torch.ones((n,horizon)), constraint=positive)

        for i in range(n):
            pyro.sample(f"x_{i}", MultivariateNormal(x_mu[i], torch.diag(x_std[i])+1e-6*torch.eye(horizon)))

        for i in range(m):
            pyro.sample(f"w_{i}", Normal(w_mu[i], w_std[i]))

    # nuts_kernel = NUTS(model, jit_compile=1)
    # mcmc = MCMC(
    #     nuts_kernel,
    #     num_samples=1000,
    #     warmup_steps=1000,
    #     num_chains=1,
    # )
    # mcmc.run(y.float())
    # print(mcmc.summary(prob=0.9))

    adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))

    losses = []
    n_steps = 2000
    # do gradient steps
    for step in range(n_steps):
        loss = svi.step(y.float())
        losses.append(loss)
        if step % 50 == 0:
            print("Elbo loss: {} for {}/{}".format(loss, step, n_steps))

    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.show()

    est_slope = pyro.param("w_mu").data.cpu().numpy()
    plt.scatter(est_slope, true_slope)
    plt.show()

    est_x = pyro.param("x_mu").data.cpu().numpy()
    plt.scatter(est_x, true_x)
    plt.show()

def test():
    data = []
    for _ in range(6):
        data.append(torch.tensor(1.0))
    for _ in range(4):
        data.append(torch.tensor(0.0))

    def model(data):
        # define the hyperparameters that control the Beta prior
        alpha0 = torch.tensor(10.0)
        beta0 = torch.tensor(10.0)
        # sample f from the Beta prior
        f = pyro.sample("latent_fairness", pyro.distributions.Beta(alpha0, beta0))
        # loop over the observed data
        for i in range(len(data)):
            # observe datapoint i using the ernoulli likelihood
            pyro.sample("obs_{}".format(i), pyro.distributions.Bernoulli(f), obs=data[i])

    def guide(data):
        # register the two variational parameters with Pyro
        # - both parameters will have initial value 15.0.
        # - because we invoke constraints.positive, the optimizer
        # will take gradients on the unconstrained parameters
        # (which are related to the constrained parameters by a log)
        alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                            constraint=positive)
        beta_q = pyro.param("beta_q", torch.tensor(15.0),
                            constraint=positive)
        # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
        pyro.sample("latent_fairness", pyro.distributions.Beta(alpha_q, beta_q))

    # setup the optimizer
    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    for step in range(1000):
        svi.step(data)
        if step % 100 == 0:
            print('.', end='')

    # grab the learned variational parameters
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    # here we use some facts about the Beta distribution
    # compute the inferred mean of the coin's fairness
    inferred_mean = alpha_q / (alpha_q + beta_q)
    # compute inferred standard deviation
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * np.sqrt(factor)

    print("\nBased on the data and our prior belief, the fairness " +
        "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

if __name__ == "__main__":
    main()