import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from scipy.stats import norm
from pymc import HalfCauchy, Model, Normal, sample
from pymc.gp.util import plot_gp_dist

def main():
    data = pd.read_csv("./data/item_long_unimp.csv")
    # SID: the subject ID
    # all_beeps: the assessment wave
    # Trait: one of the Big Five
    # Facet: one of the facets hypothesized to be under one of the Big Five traits (3 of each big 5)
    # Item: the item name (4 per facet)
    # Value: the rating 

    # right now focus on year 2018 and exclude year 2019
    data = data[data.all_beeps<=2000]
    SIDs = data.SID.unique()
    Items = data.Item.unique()
    horizon = data.all_beeps.unique()
    horizon.sort()
    C = max(data.value.unique())

def test():
    n = 200
    m = 100
    horizon = 50
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    true_slope = np.concatenate((np.random.normal(2,0.5,size=(m//2,1)),
            np.random.normal(-2,0.5,size=(m//2,1))), axis=0).reshape(m,1)
    
    # draw dynamic latent trait from gp
    # x = np.zeros((n, horizon))
    # ts = np.arange(horizon)
    # for i in range(n):
    #     # The latent function values are one sample from a multivariate normal
    #     # Note that we have to call `eval()` because PyMC built on top of Theano
    #     f_true = pm.draw(pm.MvNormal.dist(mu=pm.gp.mean.Zero(X), 
    #                                       cov=cov_func(X)), 1, random_seed=rng)


    x = np.concatenate((np.linspace(-2, -1, n//2),
            np.linspace(1, 2, n//2))).reshape(n,1)
    true_p = norm.cdf(np.dot(x, np.transpose(true_slope)))
    y = np.random.binomial(1, true_p)
    print(y.shape)
    
    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        slopes = pm.Normal("slopes", mu=0, sigma=2, shape=(1,m))
        xs = pm.Normal("x", mu=0, sigma=1, shape=(n,1))
        p = pm.math.dot(xs,slopes)
        print(p.shape)
        p = (pm.math.erf(p/np.sqrt(2))+1)/2
        # Define likelihood
        likelihood = pm.Bernoulli("y", p=p, observed=y)
        # Inference!
        # draw 1000 posterior samples using NUTS sampling
        idata = sample(1000, tune=1000, chains=1, cores=1, nuts_sampler="numpyro")

    # az.plot_trace(idata, figsize=(10, 7),var_names="slopes",
    #               coords={"slopes_dim_0": 1-1, "slopes_dim_1": 1-1})
    # plt.show()

    est_slopes = idata.posterior.slopes[0,:,0,:].mean(axis=0).to_numpy()
    plt.scatter(est_slopes, true_slope)
    plt.show()

    X = idata.posterior.x[0,:,:,0].mean(axis=0).to_numpy()
    plt.scatter(x, X)
    plt.show()

def test2():
    n = 50
    m = 60
    horizon = 100
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    true_slope = np.concatenate((np.random.normal(2,0.5,size=(m//2,horizon)),
            np.random.normal(-2,0.5,size=(m//2,horizon))), axis=0).reshape(m,horizon).T
    
    # draw dynamic latent trait from gp
    ts = np.arange(horizon)[:,None]
    X_gp = np.dstack(np.meshgrid(ts.squeeze(1), np.arange(n))).reshape(-1, 2)
    print(X_gp.shape)

    # The latent trait trends are one sample from a multivariate normal
    mean_func = pm.gp.mean.Zero()
    t_cov_func = 2**2*pm.gp.cov.ExpQuad(2, horizon//5, active_dims=[0])
    i_cov_func = pm.gp.cov.ExpQuad(2,0.01, active_dims=[1])
    x = pm.draw(pm.MvNormal.dist(mu=mean_func(X_gp), 
                cov=t_cov_func(X_gp)*i_cov_func(X_gp)), 1, random_seed=rng)

    stacked_slopes = pm.math.concatenate([true_slope for _ in range(n)], axis=0)
    true_p = pm.math.invlogit(pm.math.stack([x for _ in range(m)],axis=1) * stacked_slopes)
    y = pm.Bernoulli.dist(true_p).eval()
    print(y.shape)

    with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        slopes = pm.Normal("slopes", mu=0, sigma=2, shape=(horizon,m))

        # i.i.d gp trend for each unit
        t_cov_func = 2**2*pm.gp.cov.ExpQuad(2, horizon//5, active_dims=[0])
        i_cov_func = pm.gp.cov.ExpQuad(2,0.01, active_dims=[1])
        gp = pm.gp.Latent(cov_func=i_cov_func*t_cov_func)
        f = gp.prior("f", X_gp)

        # i.i.d standard normal for each time/unit
        # xs = pm.Normal("x", mu=0, sigma=1, shape=(n,horizon))
        # p = pm.math.stack([xs for _ in range(m)], axis=1) * pm.math.stack([slopes for _ in range(n)], axis=0)
        p = pm.math.stack([f for _ in range(m)], axis=1) * pm.math.concatenate([slopes for _ in range(n)], axis=0)
        p = pm.Deterministic("p", pm.math.invlogit(p))
        # Define likelihood
        likelihood = pm.Bernoulli("y", p=p, observed=y)
        # Inference!
        # draw 1000 posterior samples using NUTS sampling
        idata = sample(1000, tune=1000, chains=1, cores=1, nuts_sampler="numpyro")

    f_post = az.extract(idata, var_names=["slopes","f"]).transpose("sample", ...)

    fig = plt.figure(figsize=(10, 4))
    
    est_slopes = idata.posterior.slopes[0,:,:,:].mean(axis=0).to_numpy()
    plt.scatter(est_slopes.reshape(-1,1), true_slope.reshape(-1,1))
    plt.savefig("./results/scatter_w.pdf")
    plt.show()

    X = idata.posterior.f[0,:,:].mean(axis=0).to_numpy()
    plt.scatter(x, X)
    plt.savefig("./results/scatter_x.pdf")
    plt.show()

def latent_gp():
    RANDOM_SEED = 8998
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")

    n = 100  # The number of data points
    X = np.linspace(0, 10, n)[:, None]  # The inputs to the GP must be arranged as a column vector

    # Define the true covariance function and its parameters
    ell_true = 2.0
    eta_true = 4.0
    cov_func = eta_true**2 * pm.gp.cov.ExpQuad(1, ell_true)

    # A mean function that is zero everywhere
    mean_func = pm.gp.mean.Zero()

    # The latent function values are one sample from a multivariate normal
    # Note that we have to call `eval()` because PyMC built on top of Theano
    f_true = pm.draw(pm.MvNormal.dist(mu=mean_func(X), cov=cov_func(X)), 1, random_seed=rng)

    # The observed data is the latent function plus a small amount of T distributed noise
    p_true = pm.math.invlogit(f_true)
    y = pm.Bernoulli.dist(p_true).eval()

    with pm.Model() as model:
        cov = 4**2 * pm.gp.cov.ExpQuad(1, 2)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=X)
        p = pm.Deterministic("p", pm.math.invlogit(f))
        y_ = pm.Bernoulli("y", p=p, observed=y)

        idata = pm.sample(1000, tune=1000, chains=1, cores=1, nuts_sampler="numpyro")

    f_post = az.extract(idata, var_names="p").transpose("sample", ...)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()

    plot_gp_dist(ax, f_post, X)

    # plot the data and the true latent function
    ax.plot(X, pm.math.invlogit(f_true).eval(), "dodgerblue", lw=3, label="True generating function 'f'")
    ax.plot(X, y, "ok", ms=3, label="Observed data")

    # axis labels and title
    plt.xlabel("X")
    plt.ylabel("True f(x)")
    plt.title("Posterior distribution over $f(x)$ at the observed values")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test2()
