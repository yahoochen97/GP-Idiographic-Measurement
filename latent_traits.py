# loading packages
import numpy as np
import torch
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
from utilities.util import plot_task_kernel

def main(load_batch_size = 1024, num_inducing = 1000, num_epochs = 5, item_rank=5, model_type="pop"):
    # load data
    print("loading data...")
    data = pd.read_csv("./data/item_long_unimp.csv")

    # SID: the subject ID
    # all_beeps: the assessment wave
    # Trait: one of the Big Five
    # Facet: one of the facets hypothesized to be under one of the Big Five traits (3 of each big 5)
    # Item: the item name (4 per facet)
    # Value: the rating 

    data = data[(data.all_beeps<=2000) & (data.SID<=2000)]
    SIDs = data.SID.unique()
    SIDs.sort()
    Items = data.Item.unique()
    Items.sort()
    Beeps = data.all_beeps.unique()
    Beeps.sort()
    TRAITS = data.Trait.unique()
    C = max(data.value.unique())

    n = len(SIDs)
    m = len(Items)
    horizon = len(Beeps)
    Q = len(TRAITS)

    # x: [i,j,h]
    train_x = torch.zeros((data.shape[0],3))
    train_y = torch.zeros((data.shape[0],))

    iter = 0
    for _, row in data.iterrows():
        i = np.where(row["SID"]==SIDs)[0][0]
        j = np.where(row["Item"]==Items)[0][0]
        h = np.where(row["all_beeps"]==Beeps)[0][0]
        train_x[iter, 0] = i
        train_x[iter, 1] = j
        train_x[iter, 2] = h
        train_y[iter] = row["value"] - 1
        iter += 1

    print("splitting data for training/testing...")

    train_ratio = 0.8
    train_mask = np.zeros((data.shape[0],))
    train_mask[np.random.choice(data.shape[0], int(data.shape[0]*train_ratio),replace=False)] = 1
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
    # likelihood = MyDirichletClassificationLikelihood(train_y.long(), C, learn_additional_noise=True)
    # likelihood = OrdinalLikelihood(thresholds=torch.tensor([-5.,\
    #                                 -2.,-1.,1.,2.,5.]))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = OrdinalLMC(inducing_points,n,m,C,rank=item_rank, model_type=model_type)

    model.train()
    likelihood.train()

    model.t_covar_module.lengthscale = 60
    model.fixed_module.raw_lengthscale.requires_grad = False
    if model_type=="both":
        model.task_weights_module.weights = 0.5*torch.ones((n))
        model.task_weights_module.raw_weights.requires_grad = True
        final_params = list(set(model.parameters()) - \
                    {model.fixed_module.raw_lengthscale}) + \
            list(likelihood.parameters())
    else:
        model.task_weights_module.raw_weights.requires_grad = False
        if model_type=="pop":
            model.task_weights_module.weights = torch.ones((n))
            # for j in range(n):
            #     model.unit_task_covar_module[j].raw_var.requires_grad = False
            #     model.unit_task_covar_module[j].covar_factor.requires_grad = False
            
            final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale,\
                        model.task_weights_module.raw_weights}) + \
                       list(likelihood.parameters())
            #     {'params': list(set(model.parameters()) - \
            #             {model.fixed_module.raw_lengthscale,\
            #             model.task_weights_module.raw_weights})}, # - \
            #     #         {model.unit_task_covar_module[i].raw_var for i in range(n)} -\
            #     #        {model.unit_task_covar_module[i].covar_factor for i in range(n)})},
            #     {'params': likelihood.parameters()},
            # ]
        elif model_type=="ind":
            model.task_weights_module.weights = torch.zeros((n))
            model.pop_task_covar_module.raw_var.requires_grad = False
            model.pop_task_covar_module.covar_factor.requires_grad = False
            final_params = list(set(model.parameters()) - \
                        {model.fixed_module.raw_lengthscale,\
                        model.task_weights_module.raw_weights,\
                        model.pop_task_covar_module.raw_var,
                        model.pop_task_covar_module.covar_factor}) + \
                        list(likelihood.parameters())
            # [
            #     {'params': list(set(model.parameters()) - \
            #             {model.fixed_module.raw_lengthscale,\
            #             model.task_weights_module.raw_weights,\
            #             model.pop_task_covar_module.raw_var,
            #             model.pop_task_covar_module.covar_factor})},
            #     {'params': likelihood.parameters()},
            # ]

    optimizer = torch.optim.Adam(final_params, lr=0.1)

    # Our loss object. We're using the VariationalELBO
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    print("start training...")

    # model.load_state_dict(torch.load("./results/" + model_type + "_model_rank_" + str(item_rank) + ".pth"))
    # likelihood.load_state_dict(torch.load("./results/" + model_type + "_likelihood_rank_" + str(item_rank) + ".pth"))

    for i in range(num_epochs):
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            # _, transformed_target_batch = likelihood._prepare_targets(y_batch.long(),\
            #                          C,alpha_epsilon=likelihood.alpha_epsilon)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))

    torch.save(model.state_dict(), "./results/" + model_type + "_model_rank_" + str(item_rank) + ".pth")
    torch.save(likelihood.state_dict(), "./results/" + model_type + "_likelihood_rank_" + str(item_rank) + ".pth")

    print("in-sample evaluatiion...")
    model.eval()
    likelihood.eval()
    means = torch.tensor([0])
    true_ys = torch.tensor([0])
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for x_batch, y_batch in train_loader:
            test_dist = likelihood(model(x_batch))
            # probabilities = test_dist.probs
            probabilities = test_dist.loc
            means = torch.cat([means, probabilities])
            true_ys = torch.cat([true_ys, y_batch])
    means = means[1:]
    true_ys = true_ys[1:]
    # means = means.argmax(dim=0)
    # means = torch.round(means)

    # train_acc = torch.sum(true_ys==means) / true_ys.shape[0]
    train_acc1 = torch.sum(torch.abs(true_ys-means)<=0.5) / true_ys.shape[0]
    # print(train_acc)
    print(train_acc1)

    print("out-of-sample evaluatiion...")

    means = torch.tensor([0])
    true_ys = torch.tensor([0])
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for x_batch, y_batch in test_loader:
            test_dist = likelihood(model(x_batch))
            # probabilities = test_dist.probs
            probabilities = test_dist.loc
            means = torch.cat([means, probabilities])
            true_ys = torch.cat([true_ys, y_batch])
    means = means[1:]
    true_ys = true_ys[1:]

    # test_acc = torch.sum(true_ys==means) / true_ys.shape[0]
    test_acc1 = torch.sum(torch.abs(true_ys-means)<=0.5) / true_ys.shape[0]

    # print(test_acc)
    print(test_acc1)

    import os
    directory = "./results/" +  model_type
    if not os.path.exists(directory):
        os.makedirs(directory)

    if model_type=="pop":
        task_kernel = model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
        file_name = "./results/" + model_type + "/" + model_type + "_rank_{}_task.pdf".format(item_rank)
        plot_task_kernel(task_kernel, Items, file_name)
    else:
        weights = model.task_weights_module.weights.detach().numpy()
        for i in range(n):
            file_name = "./results/" + model_type + "/" + model_type + "_rank_{}_unit_{}_task.pdf".format(item_rank, i)
            task_kernel = np.zeros((m,m))
            if model_type=="both":
                task_kernel += weights[i] * model.pop_task_covar_module.covar_matrix.evaluate().detach().numpy()
            task_kernel += (1-weights[i]) * model.unit_task_covar_module[i].covar_matrix.evaluate().detach().numpy()
            plot_task_kernel(task_kernel, Items, file_name)
        
    # plt.figure(figsize=(12, 10))
    # sns.heatmap((model.unit_covar_module.covar_matrix.evaluate()).detach().numpy(),\
    #             cmap=sns.cm.rocket_r)
    # plt.savefig("./results/unit_rank_1.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-r item_rank -t model_type')
    parser.add_argument('-r','--item_rank', help='rank of task kernels', required=False)
    parser.add_argument('-t','--model_type', help='type of models pop, ind or both', required=False)
    args = vars(parser.parse_args())
    load_batch_size = 1024
    num_inducing = 1000
    num_epochs = 5
    if "item_rank" not in args:
        item_rank = 5
    else:
        item_rank= int(args["item_rank"])
    if "model_type" in args:
        model_type = args["model_type"]
    else:
        model_type = "pop" # "ind", "both"
    main(load_batch_size, num_inducing, num_epochs, item_rank, model_type)