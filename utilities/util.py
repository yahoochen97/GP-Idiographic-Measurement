import pandas as pd
import torch
import gpytorch
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.module import Module
from gpytorch.mlls import KLGaussianAddedLossTerm
from torch.distributions import MultivariateNormal
from typing import Optional, Tuple

#!/usr/bin/env python3

import warnings

import torch

from gpytorch.distributions import base_distributions
from gpytorch.functions import log_normal_cdf
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import NormalPrior
from torch.nn import ModuleList

def load_data():
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
    return data

class DynamicLatentVariable(Module):
    """
    This super class is used to describe the type of inference
    used for the latent variable :math:`\\mathbf X` in GPLVM models.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    """

    def __init__(self, n, horizon, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim
        self.horizon = horizon

    def forward(self, x):
        raise NotImplementedError
    
class PointDynamicLatentVariable(DynamicLatentVariable):
    """
    This class is used for GPLVM models to recover a MLE estimate of
    the latent variable :math:`\\mathbf X`.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    """

    def __init__(self, n, horizon, latent_dim, X_init):
        super().__init__(n, horizon, latent_dim)
        self.register_parameter("X", X_init)

    def forward(self):
        return self.X
    
class MAPDynamicLatentVariable(DynamicLatentVariable):
    """
    This class is used for GPLVM models to recover a MAP estimate of
    the latent variable :math:`\\mathbf X`, based on some supplied prior.

    :param int n: Size of the latent space.
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization for the point estimate of :math:`\\mathbf X`
    :param ~gpytorch.priors.Prior prior_x: prior for :math:`\\mathbf X`
    """

    def __init__(self, n, horizon, latent_dim, X_init, prior_x):
        super().__init__(n, horizon, latent_dim)
        self.prior_x = prior_x
        self.register_parameter("X", X_init)
        self.register_prior("prior_x", prior_x, "X")

    def forward(self):
        return self.X

    
class VariationalDynamicLatentVariable(DynamicLatentVariable):
    """
    This class is used for dynamic GPLVM models to recover a variational approximation of
    the latent variable :math:`\\mathbf X`.

    :param int n: Size of the latent space.
    :param int data_dim: Dimensionality of the :math:`\\mathbf Y` values.
    :param int horizon: length of length series
    :param int latent_dim: Dimensionality of latent space.
    :param torch.Tensor X_init: initialization of :math:`\\mathbf X`, shape of n x (latent_dim*horizon)
    :param ~gpytorch.priors.Prior prior_x: prior for :math:`\\mathbf X`, shape of n x (latent_dim*horizon)
    """

    def __init__(self, n, data_dim, horizon, latent_dim, X_init, prior_x):
        super().__init__(n, horizon, latent_dim)
        self.data_dim = data_dim # m
        self.prior_x = prior_x

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init) # n x (horizon*laten_dim)
        # K_dim = diag(v) + BB^T (v>0)
        self.q_dim_v = torch.nn.Parameter(torch.randn(latent_dim,1)) 
        self.q_dim_B = torch.nn.Parameter(torch.randn(latent_dim,1)) 
        self.q_time_var = torch.nn.Parameter(torch.ones(n,)) # diagonal K_t
        # This will add the KL divergence KL(q(X_i) || p(X_i)) to the loss
        for i in range(n):
            self.register_added_loss_term(f"x_{i}_kl")

    def forward(self):
        # Variational distribution over the latent variable q(x)
        results = torch.zeros((self.n, self.horizon*self.latent_dim))
        for i in range(self.n):
            # sigmoid for v so they are in the range of [0,1]
            # 2 sigmoid - 1 for B so they are in the range of [-1,1]
            tmp = 2*torch.sigmoid(self.q_dim_B) - 1
            K_dim = torch.matmul(tmp, tmp.t()) + \
                    torch.diag(torch.sigmoid(self.q_dim_v))
            # sigmoid for q_var so the variational variance are between 0 and 1
            K_t = torch.sigmoid(self.q_time_var[i]) * torch.eye(self.horizon)
            q_x = MultivariateNormal(self.q_mu[i], torch.kron(K_dim, K_t)\
                                     +1e-6*torch.eye(self.horizon*self.latent_dim))
            x_kl = KLGaussianAddedLossTerm(q_x, self.prior_x, 1, self.data_dim)
            self.update_added_loss_term(f"x_{i}_kl", x_kl)  # Update the KL term
            results[i] = q_x.rsample()
        return results

class ConstantVectorMean(Mean):
    def __init__(self, size, prior=None, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector",\
                 parameter=torch.nn.Parameter(torch.zeros(*batch_shape, *size)))
        if prior is not None:
            self.register_prior("constantvector_prior", prior, "constantvector")

    def forward(self, input):
        results = torch.zeros((input.shape[0],))
        for i in range(input.shape[0]):
            results[i] = self.constantvector[input[i,0].int(),input[i,1].int()]
        return results
    
class MaskMean(Mean):
    def __init__(
        self,
        base_mean: Mean,
        batch_shape: torch.Size = torch.Size(),
        active_dims: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__()
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.active_dims = active_dims
        self.base_mean = base_mean
        self.batch_shape = batch_shape
    
    def forward(self, x, **params):
        return self.base_mean.forward(x.index_select(-1, self.active_dims), **params)

class LinearMeanWithPrior(Mean):
    def __init__(self, input_size, weights_prior=None, weights_constraint=None, batch_shape=torch.Size()):
        super().__init__()
        self.register_parameter(name="raw_weights", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size)))
        if weights_prior is not None:
            self.register_prior("weights_prior", weights_prior, self._weights_param, self._weights_closure)
        if weights_constraint is not None:
            self.register_constraint("raw_weights", weights_constraint)

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        return res
    
    @property
    def weights(self):
        return self._weights_param(self)

    @weights.setter
    def weights(self, value):
        return self._weights_closure(self, value)

    def _weights_param(self, m):
        if hasattr(m, "raw_weights_constraint"):
            return m.raw_weights_constraint.transform(m.raw_weights)
        return m.raw_weights

    def _weights_closure(self, m, values):
        if not torch.is_tensor(values):
            value = torch.as_tensor(values).to(m.raw_weights)

        if hasattr(m, "raw_weights_constraint"):
            m.initialize(raw_weights=m.raw_weights_constraint.inverse_transform(values))
        else:
            m.initialize(raw_weights=values)

from gpytorch.constraints import Positive

class OrdinalLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Ordinal likelihood used for GP classification, using
    logit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF :math:`\Phi(x)`).
    .. math::
        \begin{equation*}
            p(Y=c|f)=\Phi(b_c-f)-\Phi(b_{c-1}-f)
        \end{equation*}
    """
    has_analytic_marginal: bool = True

    def __init__(self, thresholds, **kwargs):
        super().__init__(**kwargs)
        self.C = thresholds.shape[0] - 1
        thresholds, _ = torch.sort(thresholds)
        self.register_parameter("b0", parameter=torch.nn.Parameter(\
                torch.Tensor(thresholds[0]), requires_grad=True))
        self.register_parameter("deltas", parameter=torch.nn.Parameter(\
                self._threshold_to_delta(thresholds), requires_grad=True))
        self.register_constraint("deltas", Positive())
        
    def _threshold_to_delta(self, thresholds):
        C = self.C
        deltas = torch.zeros((C,))
        # deltas[0] = thresholds[1]
        for c in range(C):
            deltas[c] = thresholds[c+1] - thresholds[c]
        return deltas
    
    def _delta_to_threshold(self):
        # C = self.C
        # thresholds = torch.zeros((C+1,))
        # thresholds[0] = self.b0
        # for c in range(C):
        #     thresholds[c+1] = thresholds[c] + self.deltas[c]
        return torch.cumsum(torch.cat([torch.tensor([self.b0]),self.deltas]), dim=0)
    
    def _get_thresholds(self):
        return self._delta_to_threshold()

    def forward(self, function_samples, **kwargs):
        thresholds = self._get_thresholds()
        link2 = thresholds[1:] - function_samples
        link1 = thresholds[:-1] - function_samples
        norm_func = base_distributions.Normal(0, 1).cdf
        output_probs = norm_func(link2) - norm_func(link1)
        return base_distributions.Categorical(probs=output_probs.t())

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)
    
    def marginal(self, function_dist, **kwargs):
        mean = function_dist.mean
        var = function_dist.variance
        thresholds = self._get_thresholds()
        link2 = (thresholds[1:].unsqueeze(1).repeat(torch.Size([1,*mean.shape]))\
                  - mean).div(torch.sqrt(1+var))
        link1 = (thresholds[:-1].unsqueeze(1).repeat(torch.Size([1,*mean.shape]))\
                  - mean).div(torch.sqrt(1+var))
        norm_func = base_distributions.Normal(0, 1).cdf
        output_probs = norm_func(link2) - norm_func(link1)
        return base_distributions.Categorical(probs=output_probs.t())
    
    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        observations = observations.long()
        thresholds = self._get_thresholds()
        norm_func = base_distributions.Normal(0, 1).cdf

        log_prob_lambda = lambda function_samples: (norm_func(thresholds[observations[0]] - function_samples) \
                  - norm_func(thresholds[observations[0]-1]- function_samples) + 1e-10).log()
        log_prob = self.quadrature(log_prob_lambda, function_dist[0])
        for i in range(1,observations.size(0)):
            log_prob_lambda = lambda function_samples: (norm_func(thresholds[observations[i]] - function_samples) \
                  - norm_func(thresholds[observations[i]-1]- function_samples) + 1e-10).log()
            log_prob += self.quadrature(log_prob_lambda, function_dist[i])
        return log_prob
    
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from copy import deepcopy
from torch import Tensor

class MyDirichletClassificationLikelihood(FixedNoiseGaussianLikelihood):
    """
    A classification likelihood that treats the labels as regression targets with fixed heteroscedastic noise.
    From Milios et al, NeurIPS, 2018 [https://arxiv.org/abs/1805.10915].

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param targets: classification labels.
    :type targets: torch.Tensor (N).
    :param alpha_epsilon: tuning parameter for the scaling of the likeihood targets. We'd suggest 0.01 or setting
        via cross-validation.
    :type alpha_epsilon: int.

    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    Example:
        >>> train_x = torch.randn(55, 1)
        >>> labels = torch.round(train_x).long()
        >>> likelihood = DirichletClassificationLikelihood(targets=labels, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 1)
        >>> test_labels = torch.round(test_x).long()
        >>> pred_y = likelihood(gp_model(test_x), targets=labels)
    """

    def _prepare_targets(self, targets, num_classes, alpha_epsilon=0.01, dtype=torch.float):
        # set alpha = \alpha_\epsilon
        alpha = alpha_epsilon * torch.ones(targets.shape[-1], num_classes, device=targets.device, dtype=dtype)

        # alpha[class_labels] = 1 + \alpha_\epsilon
        alpha[torch.arange(len(targets)), targets] = alpha[torch.arange(len(targets)), targets] + 1.0

        # sigma^2 = log(1 / alpha + 1)
        sigma2_i = torch.log(1 / alpha + 1.0)

        # y = log(alpha) - 0.5 * sigma^2
        transformed_targets = alpha.log() - 0.5 * sigma2_i

        return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype)

    def __init__(
        self,
        targets: Tensor,
        num_classes: int = 2,
        alpha_epsilon: int = 0.01,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        dtype: Optional[torch.dtype] = torch.float,
        **kwargs,
    ):
        sigma2_labels, transformed_targets = self._prepare_targets(
            targets, num_classes, alpha_epsilon=alpha_epsilon, dtype=dtype
        )
        super().__init__(
            noise=sigma2_labels,
            learn_additional_noise=learn_additional_noise,
            batch_shape=torch.Size((num_classes,)),
            **kwargs,
        )
        self.transformed_targets = transformed_targets.transpose(-2, -1)
        self.num_classes = num_classes
        self.targets = targets
        self.alpha_epsilon = alpha_epsilon

    def get_fantasy_likelihood(self, **kwargs):
        # we assume that the number of classes does not change.

        if "targets" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `targets` kwarg")

        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_targets = kwargs.get("noise")
        new_noise, new_targets, _ = fantasy_liklihood._prepare_targets(new_targets, self.alpha_epsilon)
        fantasy_liklihood.targets = torch.cat([fantasy_liklihood.targets, new_targets], -1)

        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])

        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def __call__(self, *args, **kwargs):
        if "targets" in kwargs:
            targets = kwargs.pop("targets")
            dtype = self.transformed_targets.dtype
            new_noise, _, _ = self._prepare_targets(targets, dtype=dtype)
            kwargs["noise"] = new_noise
        return super().__call__(*args, **kwargs)

from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import IndexKernel, RBFKernel

class WeightKernel(Kernel):
    def __init__(self, input_size, \
                #  pop_kernel, ind_kernels, unit_dim = 0, task_dim = 1, \
                 weights_constraint=Interval(0,1), batch_shape=torch.Size()):
        super().__init__()
        # weights for populational and indivual covariance, range [0,1]
        self.register_parameter(name="raw_weights", \
                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, input_size),requires_grad=True))
        if weights_constraint is not None:
            self.register_constraint("raw_weights", weights_constraint)
        # self.pop_kernel = pop_kernel
        # self.ind_kernels = ind_kernels
        # self.unit_dim = unit_dim
        # self.task_dim = task_dim

    def forward(self, x1, x2, **kwargs):
        res = (x1.reshape(-1,1)==x2.reshape(1,-1))
        res = self.weights[x1.long()] * res
        return res
    
    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        return self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weights)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

class UnitMaskKernel(Kernel):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x1, x2, **kwargs):
        res = (x1.reshape(-1,1)==x2.reshape(1,-1))
        res = (x1==self.n) * res
        return res
    
class OrdinalLMC(ApproximateGP):
    def __init__(self, inducing_points, n, m, C, horizon, pop_rank=5, unit_rank=1, model_type="pop"):
        self.C = C # cardinality of responses
        self.n = n # number of respondents
        self.m = m # number of items
        self.horizon = horizon # number of time periods
        self.model_type = model_type
        self.batch_shape = torch.Size([])

        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(inducing_points.size(0))
        q_f = VariationalStrategy(self, inducing_points, q_u, \
                                  learn_inducing_locations=False)
        super().__init__(q_f)

        # zero mean function
        self.mean_module = gpytorch.means.ZeroMean()
        # time covariance 
        self.t_covar_module = ModuleList([RBFKernel(active_dims=[2]) for i in range(n)])
        # populational item covariance
        self.pop_task_covar_module = IndexKernel(num_tasks=m, rank=pop_rank,\
                                prior=NormalPrior(0.,4.))
        # for r in range(rank):
        #     self.pop_task_covar_module.register_constraint(\
        #         self.pop_task_covar_module.covar_factor[(r*m//rank):(r*m//rank+m//rank),r], Positive())
        # individual item covaraince
        self.unit_mask_covar_module = ModuleList([UnitMaskKernel(n=i) \
                    for i in range(n)])
        if model_type=="both":
            self.unit_task_covar_module = ModuleList([IndexKernel(num_tasks=m,\
                    rank=unit_rank, prior=NormalPrior(0.,1)) for i in range(n)])
        elif model_type=="ind":
            self.unit_task_covar_module = ModuleList([IndexKernel(num_tasks=m,\
                    rank=unit_rank, prior=NormalPrior(0.,4)) for i in range(n)])
           
        # fixed matrix for indicating unit in the item task kernel
        # equals 1 if and only if unit indices are the same
        self.fixed_module = RBFKernel()
        self.fixed_module.lengthscale = 0.1
        self.fixed_module.raw_lengthscale.requires_grad = False
  
    def forward(self, x):
        # mean function
        mean_x = self.mean_module(x)
        # unit indicator
        unit_indicator_x = self.fixed_module(x[:,0])

        # task kernel
        task_covar_x = self.pop_task_covar_module(x[:,1])#.evaluate_kernel().evaluate()
        if self.model_type=="ind":
            task_covar_x *= 0
        # pop_weights = self.task_weights_module(x[:,0])
        if self.model_type!="pop":
            for i in range(self.n):
                task_covar_x += self.unit_mask_covar_module[i](x[:,0]) * \
                    (unit_indicator_x) * self.unit_task_covar_module[i](x[:,1])
           
        # product of unit indicator, task and time kernels
        covar_x = unit_indicator_x * task_covar_x

        # time kernel
        if self.horizon > 1:
            time_covar_x = self.unit_mask_covar_module[0](x[:,0]) * self.t_covar_module[0](x)
            for i in range(1, self.n):    
                time_covar_x += self.unit_mask_covar_module[i](x[:,0]) * self.t_covar_module[i](x)

            # product of time kernels
            covar_x = covar_x * time_covar_x
            
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist
    
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.patches as mpatch
plt.switch_backend('agg')
import numpy as np

def cov_to_corr(cov):
    '''
        Transform covariance matrix to correlation matrix
    '''
    Dinv = np.diag(1 / np.sqrt(np.diag(cov)))
    corr = Dinv @ cov @ Dinv
    return corr

def plot_task_kernel(task_kernel, item_names, file_name, SORT=True):
    plt.figure(figsize=(12, 10))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    if SORT:
        kernel_order = np.argsort(np.diag(task_kernel))
        task_kernel = task_kernel[kernel_order,:][:,kernel_order]
        item_names = item_names[kernel_order]

    # cov to corr
    task_kernel = cov_to_corr(task_kernel)

    colormap = "PuOr_r" 
    norm = plt.Normalize(-1.0,1.0)
    sns.heatmap(task_kernel,yticklabels=item_names, xticklabels=item_names, \
                 cmap=colormap, norm=norm)
    
    categories = [r"$\stackrel{\underbrace{\hspace{10.5em}}}{\text{Extraversion}}$",
                  r"$\stackrel{\underbrace{\hspace{10.5em}}}{\text{Agreeableness}}$",
                  r"$\stackrel{\underbrace{\hspace{10.5em}}}{\text{Conscientiousness}}$",
                  r"$\stackrel{\underbrace{\hspace{10.5em}}}{\text{Negative Emotionality}}$",
                  r"$\stackrel{\underbrace{\hspace{10.5em}}}{\text{Open-Mindedness}}$"]
    m = task_kernel.shape[0]
    for i in range(5):
        if m==45:
            plt.text(9*i+4.5, 50.3, categories[i], ha='center')
        else:
            plt.text(12*i+5.5, 67.2, categories[i], ha='center')

    categories = ["Extraversion",
                  "Agreeableness",
                  "Conscientiousness",
                  "Negative Emotionality",
                  "Open-Mindedness"]
    
    for i in range(5):
        if m==45:
            plt.text(-5.6, 9*i+4.5, categories[i], va='center', rotation=90, fontsize=10)
            plt.text(-4.9, 9*i+4.5, r"$\overbrace{\hspace{10.5em}}$", va='center', rotation=90)
        else:
            plt.text(-7.2, 12*i+5.5, categories[i], va='center', rotation=90, fontsize=10)
            plt.text(-6.4, 12*i+5.5, r"$\overbrace{\hspace{10.5em}}$", va='center', rotation=90)
    plt.savefig(file_name, bbox_inches='tight')

def agg_kernel(task_kernel):
    tmp = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            tmp[i,j] = np.mean(task_kernel[(i*9):(i*9+9),(j*9):(j*9+9)])
    return tmp

def plot_agg_task_kernel(task_kernel, pop_kernel, file_name):
    plt.figure(figsize=(12, 10))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    task_kernel = agg_kernel(task_kernel)
    pop_kernel = agg_kernel(pop_kernel)

    # cov to corr
    task_kernel = cov_to_corr(task_kernel)
    pop_kernel = cov_to_corr(pop_kernel)
    task_kernel = task_kernel - pop_kernel
    task_kernel = np.round(task_kernel, decimals=3) + 0.

    categories = ["E","A","C","N","O"]
    
    colormap = sns.diverging_palette(250, 10, sep=50, n=10, center="light", as_cmap=True)

    norm = plt.Normalize(-0.5, 0.5)
    sns.heatmap(task_kernel,xticklabels=categories, \
                yticklabels=categories, cmap=colormap, \
                norm=norm, annot=True, fmt=".3f", annot_kws={"fontsize":32})
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32) 
    plt.tick_params(left=False, bottom=False)
    plt.savefig(file_name, bbox_inches='tight')


def matrix_cluster(matrices, max_K=10):
    centroids = None
    assignments = None
    plt.figure(figsize=(12, 10))
    dists = np.zeros((matrices.shape[0], max_K))
    for k in range(1,max_K+1):
        print("running k-means for {} clusters...".format(k))
        centroids, assignments, dists_ = matrix_kmeans(matrices, k)
        dists[:,k-1] = np.sum(dists_**2)
        print("WCSS of {} means: {}".format(k, np.mean(dists[:,k-1])))
        # np.arange(1,max_K+1), 
    
    df = pd.DataFrame(data = dists, columns = [str(i) for i in range(1,max_K+1)])
    plt.plot(np.arange(1,max_K+1), np.mean(dists,axis=0))
    plt.xticks(np.arange(1,max_K+1))
    plt.ylim([0,np.max(dists)*1.1])
    plt.xlabel("num of clusters")
    plt.ylabel("within-cluster sum of square")
    plt.savefig("./results/GP_ESM/kmean_dists.pdf", bbox_inches='tight')

def matrix_kmeans(matrices, K):
    '''
        K-means algorithms for correlation matrix
        Inputs:
            - matrices: list/array of matrices
            - K: number of clustering
        Outputs:
            - centroids: list/array of centroids
            - assignments: clustering assignments
            - dists: distances to centroids
    '''
    n = matrices.shape[0]
    d = matrices.shape[1]
    dist = np.zeros(n,)
    for i in range(n):
        dist[i] = correlation_matrix_distance(matrices[i], matrices[0])
    centroids = matrices[np.argpartition(dist, -K)[-K:]]
    # np.random.seed(12345+K)
    # centroids = matrices[np.random.choice(range(n), size=K, replace=False)]

    # centroids = np.random.normal(size=(K,d,d))
    assignments = np.zeros((n,)).astype(int)
    dists = np.zeros(n,)

    while True:
        # update assignments
        new_assignments = np.zeros((n,)).astype(int)
        for i in range(n):
            dist = np.zeros(K,)
            for k in range(K):
                dist[k] = correlation_matrix_distance(matrices[i], centroids[k])
            new_assignments[i] = int(np.argmin(dist))
            dists[i] = dist[new_assignments[i]]

        if np.linalg.norm(new_assignments-assignments)==0:
            break
        else:
            assignments = new_assignments

        # update centroids
        for k in range(K):
            centroids[k] = np.mean(matrices[assignments==k], axis=0)

    return centroids, assignments, dists


def correlation_matrix_distance(r1,r2):
    return 1 - np.trace(r1 @ r2) / np.linalg.norm(r1) / np.linalg.norm(r2)

def evaluate_gpr(model, likelihood, data_loader, mll=None):
    means = torch.tensor([0])
    true_ys = torch.tensor([0])
    lls = torch.tensor([0])
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for x_batch, y_batch in data_loader:
            test_dist = likelihood(model(x_batch))
            test_dist.sample()
            if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
                probabilities = test_dist.loc
                probabilities = torch.round(np.clip(probabilities, 1, 5))
                loss = mll(test_dist, y_batch)
                lls = torch.cat([lls, loss.repeat(y_batch.shape)])
            else:
                probabilities = test_dist.probs.argmax(axis=1) + 1
                lls = torch.cat([lls, test_dist.probs[range(y_batch.shape[0]),y_batch.long()-1].log()])
            means = torch.cat([means, probabilities])
            true_ys = torch.cat([true_ys, y_batch])
    means = means[1:]
    true_ys = true_ys[1:]
    lls = lls[1:]

    acc = torch.sum(torch.abs(true_ys-means)<=0.5) / true_ys.shape[0]
    lls = torch.sum(lls) / true_ys.shape[0]
    return acc, lls