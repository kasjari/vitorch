"""
Probability utils and functions
"""
import nn as tpn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from torch.distributions.kl import register_kl
from module_distributions.radial import RadialDistribution
from module_distributions.standard_normal_prior import StandardNormalPriorDistribution


""" Likelihood models targeted for MC sample dimension tensors """
""" torch.nn losses wrapped so that they support monte carlo sample dimension """
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 mc_reduction = 'mean',
                 weight = None, 
                 size_average = None, 
                 ignore_index = -100, 
                 reduce = None,
                 reduction = 'mean'):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(weight=weight,
                                             size_average=size_average,
                                             ignore_index=ignore_index,
                                             reduce=reduce,reduction=reduction)
        if mc_reduction.lower() == "none":
            self.mc_reduction = None
        elif mc_reduction.lower() == "sum":
            self.mc_reduction = torch.sum
        elif mc_reduction.lower() == "mean":
            self.mc_reduction = torch.mean
        else:
            raise ValueError("Given mc_reduction {} is not in ['none','mean','sum']".format(mc_reduction))
            
    def forward(self, input, target):
        # input: S x B x Classes x Spatial1, x Spatial2, ...
        # target: B x 1 x Spatial1 x Spatial2, ...
        # OR
        # input: S x B x Classes
        # target: B
        sample_losses = []
        for input_ in input:
            sample_losses.append(self.base_loss(input_,target))
        sample_losses = torch.stack(sample_losses)
        if self.mc_reduction is None:
            return sample_losses
        else:
            return self.mc_reduction(sample_losses)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, 
                 mc_reduction = 'mean',
                 weight = None, 
                 size_average = None, 
                 reduce = None, 
                 reduction = 'mean',
                 pos_weight = None):
        super().__init__()
        self.base_loss = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction, 
                                              pos_weight=pos_weight)
        if mc_reduction.lower() == "none":
            self.mc_reduction = None
        elif mc_reduction.lower() == "sum":
            self.mc_reduction = torch.sum
        elif mc_reduction.lower() == "mean":
            self.mc_reduction = torch.mean
        else:
            raise ValueError("Given mc_reduction {} is not in ['none','mean','sum']".format(mc_reduction))
            
    def forward(self, input, target):
        # input: S x B x 1 x Spatial1, x Spatial2, ...
        # target: B x 1 x Spatial1 x Spatial2, ...
        # OR
        # input: S x B x Classes
        # target: B
        # BCE wants same shape for all elements. This means atleast 2d
        if target.ndim == 1:
            target = target[:,None]
        # This loss wants float targets:
        # So just incase:
        target = target.float()
        sample_losses = []
        for input_ in input:
            sample_losses.append(self.base_loss(input_,target))
        sample_losses = torch.stack(sample_losses)
        if self.mc_reduction is None:
            return sample_losses
        else:
            return self.mc_reduction(sample_losses)
        

class L1Loss(nn.Module):
    def __init__(self,
                 mc_reduction = 'mean',
                 size_average = None,
                 reduce = None, 
                 reduction = 'mean'):
        super().__init__()
        self.base_loss = nn.L1Loss(size_average=size_average, reduce=reduce, reduction=reduction)
        if mc_reduction.lower() == "none":
            self.mc_reduction = None
        elif mc_reduction.lower() == "sum":
            self.mc_reduction = torch.sum
        elif mc_reduction.lower() == "mean":
            self.mc_reduction = torch.mean
        else:
            raise ValueError("Given mc_reduction {} is not in ['none','mean','sum']".format(mc_reduction))
            
    def forward(self, input, target):
        target = target.type_as(input)
        # input: S x B x *
        sample_losses = []
        for input_ in input:
            if input_.shape != target.shape:
                target = target.view(*input_.shape)
            sample_losses.append(self.base_loss(input_,target))
        sample_losses = torch.stack(sample_losses)
        if self.mc_reduction is None:
            return sample_losses
        else:
            return self.mc_reduction(sample_losses)

class MSELoss(nn.Module):
    def __init__(self,
                 mc_reduction = 'mean',
                 size_average = None,
                 reduce = None, 
                 reduction = 'mean'):
        super().__init__()
        self.base_loss = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
        if mc_reduction.lower() == "none":
            self.mc_reduction = None
        elif mc_reduction.lower() == "sum":
            self.mc_reduction = torch.sum
        elif mc_reduction.lower() == "mean":
            self.mc_reduction = torch.mean
        else:
            raise ValueError("Given mc_reduction {} is not in ['none','mean','sum']".format(mc_reduction))
            
    def forward(self, input, target):
        # input: S x B x *
        target = target.type_as(input)
        sample_losses = []
        for input_ in input:
            if input_.shape != target.shape:
                target = target.view(*input_.shape)
            sample_losses.append(self.base_loss(input_,target))
        sample_losses = torch.stack(sample_losses)
        if self.mc_reduction is None:
            return sample_losses
        else:
            return self.mc_reduction(sample_losses)

# Generalized loss wrapper:
class MCWrapper(nn.Module):
    def __init__(self,loss,mc_reduction="mean"):
        super().__init__()
        self.base_loss = loss
        if mc_reduction.lower() == "none":
            self.mc_reduction = None
        elif mc_reduction.lower() == "sum":
            self.mc_reduction = torch.sum
        elif mc_reduction.lower() == "mean":
            self.mc_reduction = torch.mean
        else:
            raise ValueError("Given mc_reduction {} is not in ['none','mean','sum']".format(mc_reduction))
    def forward(self, input, target):
        # Input: Samples x ...
        sample_losses = []
        for input_ in input:
            sample_losses.append(self.base_loss(input_,target))
        sample_losses = torch.stack(sample_losses)
        if self.mc_reduction is None:
            return sample_losses
        else:
            return self.mc_reduction(sample_losses)
        
        




""" DIVERGENCES """
class BNN_KLD(object):
    """ Bayesian neural network (in parameters) Kullback Leibler Divergence
    
    The standard minibatch estimate of total dataset likelihood + KLD is of the form:
    sum_i=1^N loss(pred_i,y_i) + KLD ~ N/B sum_i=1^B loss(pred_i,y_i) + KLD
    So we need to scale the KLD by 1/N if we use the standard mean reduction for each mini-batch!
    loss_scaled =  1/B sum_i=1^B loss(pred_i,y_i) + 1/N KLD
    
    Notice that when logging you should scale the KLD depending on if you train or validate.
    This is because if you have BT train batches and BV validation batches and you average the results over
    the batches in one epoch, you will have:
    train_loss = 1/BT sum_i=1^BT (1/B sum_i=1^B loss(pred_i,y_i) + scale*KLD)
               = 1/NT sum_i=1^NT loss(pred_i,y_i) + scale/BT sum_i=1^BT KLD
               = 1/NT sum_i=1^NT loss(pred_i,y_i) + scale*KLD
               = 1/NT sum_i=1^NT loss(pred_i,y_i) + 1/NT*KLD
               = 1/NT (train_elbo)
    However in validation
    validation_loss = 1/NV sum_i=1^NV loss(pred_i,y_i) + 1/NT*KLD
                    = 1/NV (sum_i=1^NV loss(pred_i,y_i) + NV/NT*KLD)
                    != 1/NV (validation_elbo)
    Thus if you want to have the scaled elbo for validation as well, you need to apply different scaling (i.e. 1/NV in validation)
    """
    def __init__(self,reduction: str = "sum"):
        reduction = reduction.lower()
        if reduction not in ["none","sum","mean"]:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
        self.reduction = reduction
    
    def __call__(self,bnn,scale=np.float32(1.0)):
        klds = []
        n_params = 0
        for module in bnn.modules():
            if hasattr(module,'variational_posteriors') and hasattr(module,'priors'):
                # This is the default mode of operation in VI:
                variable_keys = list(module.variational_posteriors.keys())
                for key in variable_keys:
                    n_params += np.prod(module.variational_posteriors[key].shape)
                    kld = torch.sum(scale*kl_divergence(module.variational_posteriors[key],module.priors[key]))
                    klds.append(kld)
            elif isinstance(module,tpn.VariationalDropoutLayer):
                # If using VariationalDropoutLayer, the divergence is computed by the layer itself
                klds.append(module.KL())
        if self.reduction == "none":
            return klds
        elif self.reduction == "sum":
            return torch.sum(torch.stack(klds))
        else:
            return torch.sum(torch.stack(klds)) / n_params


def kl_divergence(p,q):
    # This function expects nn.Module similar as in the folder module_distributions.
    td_p = p.as_torch_distribution()
    td_q = q.as_torch_distribution()
    p_iterable = False
    q_iterable = False
    if hasattr(td_p,'__len__'):
        p_iterable = True
    if hasattr(td_q,'__len__'):
        q_iterable = True
    if p_iterable != q_iterable:
        raise ValueError("only one of p or q is iterable: p - {}, q - {}".format(p,q))
    if p_iterable:
        total_kl = 0.0
        for td_p_,td_q_ in zip(p_iterable,q_iterable):
            total_kl += td.kl.kl_divergence(td_p_,td_q_)
        return total_kl
    else:
        return td.kl.kl_divergence(td_p,td_q)


@register_kl(td.Normal, StandardNormalPriorDistribution)
def _kl_normal_standardnormalprior(p, q):
    # Exploit the fact that standard normal prior has zero mean and constant
    # variance in every dimension. Allow also for other than 1 sigma.
    # Use same format as torch, i.e. KLD is computed for each dimension
    # and not reduced.
    mu0,sigma0 = p.loc,p.scale
    sigma1 = q.scale
    var0 = sigma0.pow(2)
    var1 = sigma1.pow(2)
    # Use even more efficient version if sigma = 1:
    if sigma1 == 1:
        scale_ratio = var0
        mahalanobis = mu0.pow(2)
        log_scale_ratio = -2*sigma0.log()
        return 0.5*(scale_ratio + mahalanobis -1 + log_scale_ratio)
    else:
        scale_ratio = var0/var1
        mahalanobis = mu0.pow(2)/var1
        log_scale_ratio = 2*torch.log(sigma1/sigma0)
        return 0.5*(scale_ratio + mahalanobis -1 + log_scale_ratio)

@register_kl(RadialDistribution, td.Normal)
def _kl_radial_normal(p, q):
    # This function uses the same format as the Normal KLD in pytorch,
    # i.e. the KLD is computed for each element individually and not reduced
    # Using notation in the paper https://arxiv.org/pdf/1907.00865.pdf
    L_entropy = p.L_entropy()
    samples = p.rsample([p.divergence_estimation_samples])
    log_probs = q.log_prob(samples)
    L_cross_entropy = torch.mean(log_probs,0) # Average over the MC samples
    return L_entropy - L_cross_entropy

@register_kl(RadialDistribution, StandardNormalPriorDistribution)
def _kl_radial_standardnormalprior(p, q):
    # This function uses the same format as the Normal KLD in pytorch,
    # i.e. the KLD is computed for each element individually and not reduced
    # Using notation in the paper https://arxiv.org/pdf/1907.00865.pdf
    L_entropy = p.L_entropy()
    samples = p.rsample([p.divergence_estimation_samples])
    log_probs = q.log_prob(samples)
    L_cross_entropy = torch.mean(log_probs,0) # Average over the MC samples
    return L_entropy - L_cross_entropy



class BNN_ARD(object):
    """ Bayesian neural network Alpha Renyi Divergence 
    
    See the notes on KLD for scaling details
    """
    def __init__(self,alpha,reduction: str = "sum"):
        self.alpha = alpha
        if alpha == 1:
            raise ValueError("alpha parameter cannot be 1 for numerical reasons. Use Kullback Leibler Divergence instead (it is the same)")
        if alpha == 0:
            raise ValueError("alpha parameter cannot be 0 for numerical reasons. This is the negative log probability of prior where posterior is positive.")
        reduction = reduction.lower()
        if reduction not in ["none","sum","mean"]:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
        self.reduction = reduction
        
    def __call__(self,bnn,scale=np.float32(1.0)):
        ards = []
        n_params = 0
        for module in bnn.modules():
            if hasattr(module,'variational_posteriors') and hasattr(module,'priors'):
                variable_keys = list(module.variational_posteriors.keys())
                for key in variable_keys:
                    n_params += np.prod(module.variational_posteriors[key].shape)
                    ard = torch.sum(scale*ard_divergence(module.variational_posteriors[key],module.priors[key],alpha=self.alpha))
                    ards.append(ard)
        if self.reduction == "none":
            return ards
        elif self.reduction == "sum":
            return torch.sum(torch.stack(ards))
        else:
            return torch.sum(torch.stack(ards)) / n_params

""" Renyi's alpha divergence """
def ard_divergence(p,q,alpha):
    td_p = p.as_torch_distribution()
    td_q = q.as_torch_distribution()
    p_iterable = False
    q_iterable = False
    if hasattr(td_p,'__len__'):
        p_iterable = True
    if hasattr(td_q,'__len__'):
        q_iterable = True
    if p_iterable != q_iterable:
        raise ValueError("only one of p or q is iterable: p - {}, q - {}".format(p,q))
    if p_iterable:
        total_ard = 0.0
        for td_p_,td_q_ in zip(p_iterable,q_iterable):
            p_name = type(td_p_).__name__.lower().replace("distribution","")
            q_name = type(td_q_).__name__.lower().replace("distribution","")
            total_ard += eval("ard_{}_{}".format(p_name,q_name))(td_p_,td_q_,alpha=alpha)
        return total_ard
    else:
        p_name = type(td_p).__name__.lower().replace("distribution","")
        q_name = type(td_q).__name__.lower().replace("distribution","")
        return eval("ard_{}_{}".format(p_name,q_name))(td_p,td_q,alpha=alpha)


def ard_normal_normal(p, # A torch.distributions.normal.Normal class object
                      q, # A torch.distributions.normal.Normal class object
                      alpha):
    """
    https://arxiv.org/pdf/1904.02063.pdf
    GVI paper
    
    Renyi's alpha divergence is defined as (this is rescaled version):
    1/(alpha * (alpha -1)) log E_q[(p(theta) / q(theta))^(1-alpha)]
    
    The result is for normal:
    1/(2*alpha*(alpha-1))*
    [
      (1-alpha) log sigma^2
    + alpha log s^2
    - log((1-alpha)sigma^2 + alpha s^2)
    ]
    + 1/2 (mu - m)^2 / ((1-alpha)sigma^2 + alpha s^2)
    
    p = N(mu,sigma^2)
    q = N(m,s^2)
    
    For Diagonal multivariate normal, this is calculated
    independently for each dimension and then summed
    """
    mu = p.mean
    sigma_sq = p.variance
    m = q.mean
    s_sq = q.variance
    interp_var = (1.0-alpha)*sigma_sq + alpha*s_sq
    
    scale1 = 1.0/(2.0 * alpha * (alpha - 1.0))
    term1 = (1.0-alpha)*torch.log(sigma_sq) + alpha*torch.log(s_sq) - torch.log(interp_var)
    
    scale2 = 0.5
    term2 = (mu-m).pow(2)/interp_var
    
    return scale1*term1 + scale2*term2

def ard_normal_standardnormalprior(p, # A torch.distributions.normal.Normal class object
                                   q, # An md.standard_normal_prior.StandardNormalPriorDistribution class object
                                   alpha):
    """
    https://arxiv.org/pdf/1904.02063.pdf
    GVI paper
    
    Renyi's alpha divergence is defined as (this is rescaled version):
    1/(alpha * (alpha -1)) log E_q[(p(theta) / q(theta))^(1-alpha)]
    
    The result is for normal:
    1/(2*alpha*(alpha-1))*
    [
      (1-alpha) log sigma^2
    + alpha log s^2
    - log((1-alpha)sigma^2 + alpha s^2)
    ]
    + 1/2 (mu - m)^2 / ((1-alpha)sigma^2 + alpha s^2)
    
    p = N(mu,sigma^2)
    q = N(m,s^2)
    
    For Diagonal multivariate normal, this is calculated
    independently for each dimension and then summed
    """
    mu = p.mean
    sigma_sq = p.variance
    s_sq = q.variance
    interp_var = (1.0-alpha)*sigma_sq + alpha*s_sq
    
    scale1 = 1.0/(2.0 * alpha * (alpha - 1.0))
    term1 = (1.0-alpha)*torch.log(sigma_sq) + alpha*torch.log(s_sq) - torch.log(interp_var)
    
    scale2 = 0.5
    term2 = mu.pow(2)/interp_var
    
    return scale1*term1 + scale2*term2


class BNN_EP(object):
    """ Bayesian neural network (in parameters) Expectation Propagation (here treated as simply reverse KL)
    """
    def __init__(self,reduction: str = "sum"):
        reduction = reduction.lower()
        if reduction not in ["none","sum","mean"]:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
        self.reduction = reduction
    
    def __call__(self,bnn,scale=np.float32(1.0)):
        klds = []
        n_params = 0
        for module in bnn.modules():
            if hasattr(module,'variational_posteriors') and hasattr(module,'priors'):
                variable_keys = list(module.variational_posteriors.keys())
                for key in variable_keys:
                    n_params += np.prod(module.variational_posteriors[key].shape)
                    kld = torch.sum(scale*kl_divergence(module.priors[key],module.variational_posteriors[key]))
                    klds.append(kld)
        if self.reduction == "none":
            return klds
        elif self.reduction == "sum":
            return torch.sum(torch.stack(klds))
        else:
            return torch.sum(torch.stack(klds)) / n_params



@register_kl(StandardNormalPriorDistribution,td.Normal)
def _kl_standardnormalprior_normal(p, q):
    # This is mainly for EP-like KL, since the standardnormalprior
    # should not be trained
    sigma0 = p.scale
    mu1,sigma1 = q.loc,q.scale
    var0 = sigma0.pow(2)
    var1 = sigma1.pow(2)
    # Use even more efficient version if sigma = 1:
    if sigma0 == 1:
        scale_ratio = 1.0/var1
        mahalanobis = mu1.pow(2)/var1
        log_scale_ratio = 2*sigma1.log()
        return 0.5*(scale_ratio + mahalanobis -1 + log_scale_ratio)
    else:
        scale_ratio = var0/var1
        mahalanobis = mu1.pow(2)/var1
        log_scale_ratio = 2*torch.log(sigma1/sigma0)
        return 0.5*(scale_ratio + mahalanobis -1 + log_scale_ratio)

@register_kl(td.Normal,RadialDistribution)
def _kl_normal_radial(p, q):
    # Using notation in the paper https://arxiv.org/pdf/1907.00865.pdf
    # Equation 44 in the paper gives the log probability
    normal_entropy = p.entropy()
    samples = p.rsample([q.divergence_estimation_samples])
    log_probs = q.log_prob(samples)
    L_cross_entropy = torch.mean(log_probs,0) # Average over the MC samples
    return -normal_entropy - L_cross_entropy

@register_kl(StandardNormalPriorDistribution,RadialDistribution)
def _kl_standardnormalprior_radial(p, q):
    # Using notation in the paper https://arxiv.org/pdf/1907.00865.pdf
    # Equation 44 in the paper gives the log probability
    normal_entropy = p.entropy()
    samples = p.rsample([q.divergence_estimation_samples])
    log_probs = q.log_prob(samples)
    L_cross_entropy = torch.mean(log_probs,0) # Average over the MC samples
    return -normal_entropy - L_cross_entropy


"""
Bayes By Backprop estimators
These are useful if no closed form divergence can be computed for the parameters
as integration or summation is done using Monte Carlo estimation.
"""
class BNN_BBB_KLD(object):
    def __init__(self,reduction: str = "sum"):
        if reduction not in ["none","sum","mean"]:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
        self.reduction = reduction
        
    def __call__(self,bnn,scale=np.float32(1.0)):
        klds = []
        n_params = 0
        for module in bnn.modules():
            if hasattr(module,'variational_posteriors') and hasattr(module,'priors'):
                variable_keys = list(module.variational_posteriors.keys())
                for key in variable_keys:
                    n_params += np.prod(module.variational_posteriors[key].shape)
                    kld = torch.sum(scale*bbb_kl_divergence(module.variational_posteriors[key],module.priors[key]))
                    klds.append(kld)
        if self.reduction == "none":
            return klds
        elif self.reduction == "sum":
            return torch.sum(torch.stack(klds))
        else:
            return torch.sum(torch.stack(klds)) / n_params


def bbb_kl_divergence(p, q):
    """
    Computed as the bayes by backprop (Weight Uncertainty in Neural Networks)
    https://arxiv.org/pdf/1505.05424.pdf
    
    KL[p,q] = int p(x) log[p(x)/q(x)]
            ~ 1/S sum_i=1^S log[p(x_i)] - log[q(x_i)]
    """
    td_p = p.as_torch_distribution()
    td_q = q.as_torch_distribution()
    
    posterior_samples = td_p.rsample([p.divergence_estimation_samples])
    
    p_log_prob = torch.stack([td_p.log_prob(sample) for sample in posterior_samples])
    q_log_prob = torch.stack([td_q.log_prob(sample) for sample in posterior_samples])
    
    # Average over the MC samples to get monte carlo estimates of entropy and cross entropy
    return torch.mean(p_log_prob,dim=0) - torch.mean(q_log_prob,dim=0)


class BNN_BBB_ARD(object):
    """ Variational Bayesian neural network Alpha Renyi Divergence using MC integration"""
    def __init__(self,alpha,reduction: str = "sum"):
        self.alpha = alpha
        if alpha == 1:
            raise ValueError("alpha parameter cannot be 1 for numerical reasons. Use Kullback Leibler Divergence instead (it is the same)")
        if alpha == 0:
            raise ValueError("alpha parameter cannot be 0 for numerical reasons. This is the negative log probability of prior where posterior is positive.")
        reduction = reduction.lower()
        if reduction not in ["none","sum","mean"]:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
        self.reduction = reduction
        
    def __call__(self,bnn,scale=np.float32(1.0)):
        ards = []
        n_params = 0
        for module in bnn.modules():
            if hasattr(module,'variational_posteriors') and hasattr(module,'priors'):
                variable_keys = list(module.variational_posteriors.keys())
                for key in variable_keys:
                    n_params += np.prod(module.variational_posteriors[key].shape)
                    ard = torch.sum(scale*bbb_ard_divergence(module.variational_posteriors[key],module.priors[key],alpha=self.alpha))
                    ards.append(ard)
        if self.reduction == "none":
            return ards
        elif self.reduction == "sum":
            return torch.sum(torch.stack(ards))
        else:
            return torch.sum(torch.stack(ards)) / n_params


def bbb_ard_divergence(p, q, alpha):
    """
    Computed using MC integration
    We use the parametrization from https://arxiv.org/pdf/1904.02063.pdf
    D_ard[p,q] = 1/(alpha*(alpha-1))*log[int p(x)^alpha
                                             q(x)^(1-alpha)
                                             dx]
    Notice that p(x)^alpha = p(x)*p(x)^(alpha-1)
    ->
    D_ard[p,q] = 1/(alpha*(alpha-1))*log[int p(x)
                                             p(x)^(alpha-1)
                                             q(x)^(1-alpha)
                                             dx]
    D_ard[p,q] ~ log[ 1/S sum_i p(x_i)^(alpha-1)*q(x_i)^(1-alpha) ]
                 / (alpha*(alpha-1))
               = log[ 1/S sum_i exp( (alpha-1)*[log_p(x_i) - log_q(x_i)] ]
    """
    td_p = p.as_torch_distribution()
    td_q = q.as_torch_distribution()
    
    posterior_samples = td_p.rsample([p.divergence_estimation_samples])
    
    p_log_prob = torch.stack([td_p.log_prob(sample) for sample in posterior_samples])
    q_log_prob = torch.stack([td_q.log_prob(sample) for sample in posterior_samples])
    
    delta = (alpha-1)*(p_log_prob - q_log_prob)
    
    # For numerical stability we can extract the maximum term out of the sum:
    delta_max = torch.max(delta)
    shifted_deltas = delta - delta_max
    exp_term = torch.exp(shifted_deltas)
    log_argument = torch.mean(exp_term)
    
    return (delta_max + torch.log(log_argument))/(alpha*(alpha-1))


""" Batch Divergences """
class Batch_KLD(object):
    """ Kullback Leibler divergence for a batch of distributions (Like in VAE's)"""
    def __init__(self,reduction: str = "mean"):
        if reduction.lower() == "none":
            self.reduction = None
        elif reduction.lower() == "sum":
            self.reduction = torch.sum
        elif reduction.lower() == "mean":
            self.reduction = torch.mean
        else:
            raise ValueError("Given reduction {} is not in ['none','mean','sum']".format(reduction))
            
    def __call__(self,p,q,scale=np.float32(1.0)):
        # KLD[p||q]
        divergences = td.kl.kl_divergence(p,q)
        divergences = scale*torch.sum(divergences,dim=tuple(range(1,divergences.ndim)))
        if self.reduction is None:
            return divergences
        else:
            return self.reduction(divergences)
        
class Batch_None(object):
    """ None divergence. For compatability"""
    def __init__(self,*args,**kwargs):
        super(Batch_None,self).__init__()
        
    def __call__(self,p,q,scale=np.float32(1.0)):
        return 0.0









