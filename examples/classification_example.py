import sys
sys.path.insert(1, '../')
import nn as pnn
import utils
import module_distributions as md
from measures import BNN_KLD,BNN_ARD
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# For this example
from sklearn.datasets import make_moons

def entropy(p):
    p = np.clip(p,1e-6,1-1e-6)
    return -(p*np.log(p)+(1-p)*np.log(1-p))       

def clf():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    
    epochs = 1000
    n = 1000
    x,y = make_moons(n,noise=0.1)
    y = y.astype("float32")
    x = x.astype("float32")
    x = x-np.mean(x,axis=0)
    x = x/np.std(x,axis=0)
    yt = torch.from_numpy(y[:,None]).to("cuda:0")
    xt = torch.from_numpy(x).to("cuda:0")
    # Train a normal neural network:
    activation = nn.ReLU
    neurons = 100
    model = nn.Sequential(nn.Linear(2,neurons),
                          activation(),
                          nn.Linear(neurons,neurons),
                          activation(),
                          nn.Linear(neurons,1))
    model = model.to("cuda:0")
    neg_log_likelihood = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters())
    
    for i in range(epochs):
        pred = model(xt)
        nll = neg_log_likelihood(pred,yt)
        nll.backward()
        optim.step()
        optim.zero_grad()
        if i%100 == 0:
            pred = torch.sigmoid(pred).detach().cpu().numpy()>0.5
            pred = pred.astype("float32").squeeze()
            accuracy = np.mean(pred==y)
            print("Epoch: {}, accuracy: {}, NLL: {}".format(i,accuracy,nll.detach().cpu().numpy()))
    
    # Make some predictions:
    num = 50
    lim = 4
    spacex,spacey = np.meshgrid(np.linspace(-lim,lim,num=num),np.linspace(-lim,lim,num=num),indexing="ij")
    spacex_ = spacex.reshape(num**2,1)
    spacey_ = spacey.reshape(num**2,1)
    space = np.concatenate([spacex_,spacey_],axis=1)
    spacet = torch.from_numpy(space.astype("float32")).to("cuda:0")
    preds_ML = torch.sigmoid(model(spacet)).detach().cpu().numpy().reshape(num,num)
    uncertainty_ML = entropy(preds_ML)
    model = model.cpu()
    
    # Make a BNN:
    convert = False
    if not convert:
        # You can either do this:
        model = nn.Sequential(pnn.VILinear(2,neurons), # You can use gradient_variance_reduction_method to use possibly lower variance estimators
                              pnn.Wrap(activation()),
                              pnn.VILinear(neurons,neurons),
                              pnn.Wrap(activation()),
                              pnn.VILinear(neurons,1))
    else:
        # Or just use existing pytorch model and convert it to a bnn like so:
        # Notice that converted network does not support MCSampler, as activations etc. are not wrapped
        # MC samples are handled as a list in this implementation.
        model = utils.convert_to_mfvi_network(model,
                                              weight_posterior = md.Normal,#Radial,
                                              bias_posterior = md.Normal,#md.Radial,
                                              weight_prior = md.StandardNormalPrior,
                                              bias_prior = md.StandardNormalPrior,
                                              gradient_variance_reduction_method_linear="lrt")
    
    model = model.to("cuda:0")
    # If you want to use multiple MC samples, you can wrap the model into an MC sampler:
    mc_samples = 50
    model = pnn.MCSampler(model,mc_samples,stack_result=True)
    # It is then possible to modify the mc samples for example between training and validation by:
    model.mc_samples = 5
    # train loop
    model.mc_samples = 1000
    # validation loop
    # etc.
    # Lets just use the same number of samples in all phases now
    model.mc_samples = mc_samples
    
    # If you use mc samples make sure you have a loss that can be computed with either a list
    # or a tensor with an additional dimension for mc samples.
    # MCSampler returns by default a tensor with shape:
    # mc_samples,batch_size,outputdim1,outputdim2,...
    # If you set MCSampler(..., stack_result=False), it will return a list
    neg_log_likelihood = nn.BCEWithLogitsLoss()
    # BNN_KLD computes automatically the KL divergence on layers that have an approximate posterior and a prior
    divergence = BNN_KLD()
    # Set the scale so that the ELBO is valid
    scale = 1.0/n
    # You can use any optimizer
    optim = torch.optim.Adam(model.parameters())
    
    # Then just train:
    for i in range(epochs):
        pred = model(xt)
        nlls = [neg_log_likelihood(pred[j],yt) for j in range(model.mc_samples)]
        nll = torch.mean(torch.stack(nlls))
        kld = divergence(model,scale=scale)
        loss = nll + kld
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i%100 == 0:
            pred = torch.mean(pred,dim=0)
            pred = torch.sigmoid(pred).detach().cpu().numpy()>0.5
            pred = pred.astype("float32").squeeze()
            accuracy = np.mean(pred==y)
            print("Epoch: {}, accuracy: {}, NLL: {}, KLD: {}".format(i,accuracy,nll.detach().cpu().numpy(),kld.detach().cpu().numpy()))
    
    # Make some predictions:
    # For better visual quality we can change the gradient variance reduction method to none so that we get
    # an ensemble of mc_samples networks
    utils.change_gradient_variance_reduction_method(model,gradient_variance_reduction_method_linear="none")
    num = 50
    spacex,spacey = np.meshgrid(np.linspace(-lim,lim,num=num),np.linspace(-lim,lim,num=num),indexing="ij")
    spacex_ = spacex.reshape(num**2,1)
    spacey_ = spacey.reshape(num**2,1)
    space = np.concatenate([spacex_,spacey_],axis=1)
    spacet = torch.from_numpy(space.astype("float32")).to("cuda:0")
    preds_MC = torch.sigmoid(model(spacet)).detach().cpu().numpy()
    preds_PP = np.mean(preds_MC,axis=0).reshape(num,num)
    uncertainty_PP = entropy(preds_PP)
    
    
    # There is also support for alpha Renyi divergence that can be used
    # in generalized variational inference:
    model = nn.Sequential(pnn.VILinear(2,neurons,gradient_variance_reduction_method="lrt"),
                          pnn.Wrap(activation()),
                          pnn.VILinear(neurons,neurons,gradient_variance_reduction_method="lrt"),
                          pnn.Wrap(activation()),
                          pnn.VILinear(neurons,1,gradient_variance_reduction_method="lrt"))
    model = pnn.MCSampler(model,mc_samples,stack_result=True).to("cuda:0")
    neg_log_likelihood = nn.BCEWithLogitsLoss()
    # Select alpha Renyi
    divergence = BNN_ARD(alpha=0.5)
    scale = 1.0/n
    optim = torch.optim.Adam(model.parameters())
    
    # Then just train:
    for i in range(epochs):
        pred = model(xt)
        nlls = [neg_log_likelihood(pred[j],yt) for j in range(model.mc_samples)]
        nll = torch.mean(torch.stack(nlls))
        ard = divergence(model,scale=scale)
        loss = nll + ard
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i%100 == 0:
            pred = torch.mean(pred,dim=0)
            pred = torch.sigmoid(pred).detach().cpu().numpy()>0.5
            pred = pred.astype("float32").squeeze()
            accuracy = np.mean(pred==y)
            print("Epoch: {}, accuracy: {}, NLL: {}, ARD: {}".format(i,accuracy,nll.detach().cpu().numpy(),ard.detach().cpu().numpy()))
    
    # Make some predictions:
    utils.change_gradient_variance_reduction_method(model,gradient_variance_reduction_method_linear="none")
    num = 50
    spacex,spacey = np.meshgrid(np.linspace(-lim,lim,num=num),np.linspace(-lim,lim,num=num),indexing="ij")
    spacex_ = spacex.reshape(num**2,1)
    spacey_ = spacey.reshape(num**2,1)
    space = np.concatenate([spacex_,spacey_],axis=1)
    spacet = torch.from_numpy(space.astype("float32")).to("cuda:0")
    preds_MC = torch.sigmoid(model(spacet)).detach().cpu().numpy()
    preds_PP_ARD = np.mean(preds_MC,axis=0).reshape(num,num)
    uncertainty_PP_ARD = entropy(preds_PP_ARD)
    
    
    
    # Variational dropout is also possible:
    model = nn.Sequential(pnn.Wrap(nn.Linear(2,neurons)),
                          pnn.Wrap(activation()),
                          pnn.VariationalDropoutLayer(nn.Linear(neurons,neurons)),
                          pnn.Wrap(activation()),
                          pnn.VariationalDropoutLayer(nn.Linear(neurons,1)))
    model = pnn.MCSampler(model,mc_samples,stack_result=True).to("cuda:0")
    neg_log_likelihood = nn.BCEWithLogitsLoss()
    divergence = BNN_KLD()
    scale = 1.0/n
    optim = torch.optim.Adam(model.parameters())
    
    # Then just train:
    for i in range(epochs):
        pred = model(xt)
        nlls = [neg_log_likelihood(pred[j],yt) for j in range(model.mc_samples)]
        nll = torch.mean(torch.stack(nlls))
        kl = divergence(model,scale=scale)
        loss = nll + kl
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i%100 == 0:
            pred = torch.mean(pred,dim=0)
            pred = torch.sigmoid(pred).detach().cpu().numpy()>0.5
            pred = pred.astype("float32").squeeze()
            accuracy = np.mean(pred==y)
            print("Epoch: {}, accuracy: {}, NLL: {}, KL: {}".format(i,accuracy,nll.detach().cpu().numpy(),kl.detach().cpu().numpy()))
    
    # Make some predictions:
    utils.change_gradient_variance_reduction_method(model,gradient_variance_reduction_method_dropout="none")
    num = 50
    spacex,spacey = np.meshgrid(np.linspace(-lim,lim,num=num),np.linspace(-lim,lim,num=num),indexing="ij")
    spacex_ = spacex.reshape(num**2,1)
    spacey_ = spacey.reshape(num**2,1)
    space = np.concatenate([spacex_,spacey_],axis=1)
    spacet = torch.from_numpy(space.astype("float32")).to("cuda:0")
    preds_MC = torch.sigmoid(model(spacet)).detach().cpu().numpy()
    preds_PP_Dropout = np.mean(preds_MC,axis=0).reshape(num,num)
    uncertainty_PP_Dropout = entropy(preds_PP_Dropout)
    
    
    fig, axs = plt.subplots(4, 2, figsize=(10,20))
    data_pos = x[y== 1,:]
    data_neg = x[y == 0,:]
    for i in range(4):
        if i == 0:
            name = "Maximum Likelihood"
            mu = preds_ML
            ent = uncertainty_ML
        elif i == 1:
            name = "Variational BNN"
            mu = preds_PP
            ent = uncertainty_PP
        elif i == 2:
            name = "GVI BNN"
            mu = preds_PP_ARD
            ent = uncertainty_PP_ARD
        elif i == 3:
            name = "Variational Dropout"
            mu = preds_PP_Dropout
            ent = uncertainty_PP_Dropout
        
        axs[i,0].set_title(name+" mean")
        axs[i,1].set_title(name+" entropy")
        axs[i,0].contourf(spacex,spacey,mu)
        axs[i,1].contourf(spacex,spacey,ent)
        axs[i,0].scatter(data_pos[:,0],data_pos[:,1],alpha=0.5,color="red",s=1.0)
        axs[i,0].scatter(data_neg[:,0],data_neg[:,1],alpha=0.5,color="blue",s=1.0)
        axs[i,1].scatter(data_pos[:,0],data_pos[:,1],alpha=0.5,color="red",s=1.0)
        axs[i,1].scatter(data_neg[:,0],data_neg[:,1],alpha=0.5,color="blue",s=1.0)
    
    plt.savefig("../figures/demo.png")


if __name__ == '__main__':
    clf()
        
    






    
        
