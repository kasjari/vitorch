import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions import Normal as torch_Normal
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np

import math
from numbers import Real
from numbers import Number

class StandardNormalPriorDistribution(Distribution):
    
    arg_constraints = {'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return torch.zeros_like(self.scale)

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, scale, shape, validate_args=None):
        self.scale, self.shape = scale,torch.Size(shape)
        super(StandardNormalPriorDistribution, self).__init__(validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.shape
        std = self.scale.expand(shape)
        epsilon = torch.randn_like(std)
        return epsilon*std # the reparametrization trick

    def log_prob(self,input):
        std = self.scale.expand(self.shape)
        # log normalization constant: log(1/sqrt(2pi std^2))
        C = -0.5*np.log(2*np.pi)-std.log()
        # Log exponential term -1/2*(x - mu)^2/std^2
        expterm = -0.5*input.pow(2)/std.pow(2)
        return C + expterm

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        scale = self.scale.expand(self.shape)
        return 0.5 + 0.5 * np.log(2*np.pi) + scale.log()
    
    def as_normal(self):
        # Convert to torch.distributions.Normal
        scale = self.scale.expand(self.shape)
        return torch_Normal(torch.zeros_like(scale),scale)

class StandardNormalPrior(nn.Module):
    def __init__(self,
                 shape,
                 scale_requires_grad=False,
                 divergence_estimation_samples=10,
                 **kwargs):
        super(StandardNormalPrior,self).__init__()
        """
        This is meant as a prior that is identical in every dimension in variance
        with zero mean.
        It can help to reduce memory and make some divergences faster to compute.
        """
        if not hasattr(shape,'__len__'):
            shape = [shape]
        self.shape = shape
        self.scale = Parameter(torch.ones((1,)),requires_grad=scale_requires_grad)
        self.divergence_estimation_samples = divergence_estimation_samples
        
    def standard_init(self):
        # Standard normal initialization
        init.constant_(self.scale,1.0)
        
    def constant_init(self,scale_value):
        init.constant_(self.scale,scale_value)
    
    def standard_output_init(self):
        """
        Given vector w and x, h =  w^Tx where w~N(0,S), mean is 0 obviously, but
        Cov[w^Tx] = E[((Chol(S)eps)^Tx)^T((Chol(S)eps)^Tx)], where eps~N(0,I)
                  = E[x^T Chol(S) eps eps^T chol(S) x]
                  = x^TSx
                  = sum_i x_i^2 S_i
                  constant S
                  = d*s sum_i x_i^2
        If x_i^2 is 1 in expectation , then we have
        Cov[w^Tx] = d^2 s
        Thus if s = 1/d^2
        Cov[w^Tx] = 1
        This means that std = sqrt(1/d^2) = 1/d
        
        This is computed in the "fan-in" sense
        """
        fan_in = 1 if len(self.shape) == 1 else np.prod(self.shape[1:])
        init.constant_(self.scale,1.0/np.sqrt(fan_in))
    
    def custom_init(self,function):
        function(self)
    
    def freeze_scale(self):
        self.scale.requires_grad = False
    
    def unfreeze_scale(self):
        raise ValueError("Cannot train the StandardNormalPrior, use Normal instead!")
    
    def freeze_loc(self):
        return # Nothing to do here
    
    def unfreeze_loc(self):
        raise ValueError("Cannot train the StandardNormalPrior, use Normal instead!")
    
    def get_parameters(self):
        return self.scale
    
    def get_named_parameters(self):
        return {"scale": self.scale}
    
    def as_torch_distribution(self):
        scale = self.get_parameters()
        dist = StandardNormalPriorDistribution(scale=scale,shape=self.shape)
        dist.divergence_estimation_samples = self.divergence_estimation_samples
        return dist
    
    def forward(self,x):
        # Evaluates the log-pdf of the sample x
        return self.as_torch_distribution().log_prob(x)
    
    def sample(self,num_samples=None):
        dist = self.as_torch_distribution()
        if num_samples is None:
            samples = dist.rsample()
            return samples
        if not hasattr(num_samples,'__len__'):
            num_samples = [num_samples]
        num_samples = torch.Size(num_samples)
        samples = dist.rsample(num_samples)
        return samples
    
    
def kl_normal_standardnormalprior(p, q):
    # Exploit the fact that standard normal prior has zero mean and constant
    # variance in every dimension.
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
    
def unit_test():
    shape = [1,5]
    n_samples = 3
    normal1 = StandardNormalPrior(shape)
    sample = normal1.sample(n_samples)
    print("Random samples: {}".format(sample))
    print("Sample shape standard normal - correct: {}, result: {}".format(torch.Size([n_samples] + shape), sample.shape))
    
    def gaussian_log_prob(x,mu=0.0,std=1.0):
        x = x.flatten()
        if not hasattr(mu,'__len__'):
            mu = np.zeros_like(x)+mu
        if not hasattr(std,'__len__'):
            std = np.zeros_like(x)+std
        cov = np.diag(std)**2
        term1 = -len(x)/2.0*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov))
        term2 = -0.5*(x - mu)@np.linalg.pinv(cov)@(x-mu).T
        return term1+term2
    
    
    stNormal = normal1.as_torch_distribution().as_normal()
    lp_torch = torch.sum(stNormal.log_prob(sample))
    lp_mine = torch.sum(normal1(sample))
    
    sample_np = sample.detach().cpu().numpy()
    lp_np = gaussian_log_prob(sample_np)
    print("Log probability of sample - mine:{}, torch: {}, numpy: {}".format(lp_mine.item(), lp_torch.item(),lp_np))
    
    # Test if KL divergences match
    tNormal = torch_Normal(torch.randn(*shape),torch.ones(shape))
    kldiv_my = kl_normal_standardnormalprior(tNormal,normal1.as_torch_distribution())
    kldiv_torch = kl_divergence(tNormal,stNormal)
    print("KL divergence with a Normal - expecting {}, calculated {}".format(kldiv_torch,kldiv_my))
    print("KL divergence sums: expecting {}, calculated {}".format(torch.sum(kldiv_torch),torch.sum(kldiv_my)))
    
    # Test if my KL is zero for identical:
    kldiv_my = kl_normal_standardnormalprior(stNormal,normal1.as_torch_distribution())
    print("KL divergence with self - expecting {}, calculated {}".format(torch.zeros_like(kldiv_my),kldiv_my))
    
    shape = [1,5]
    n_samples = 3
    normal1 = StandardNormalPrior(shape)
    normal1.standard_output_init()
    sample = normal1.sample(n_samples)
    print("Random samples: {}".format(sample))
    print("Sample shape standard normal - correct: {}, result: {}".format(torch.Size([n_samples] + shape), sample.shape))
    
    def gaussian_log_prob(x,mu=0.0,std=1.0):
        x = x.flatten()
        if not hasattr(mu,'__len__'):
            mu = np.zeros_like(x)+mu
        if not hasattr(std,'__len__'):
            std = np.zeros_like(x)+std
        cov = np.diag(std)**2
        term1 = -len(x)/2.0*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov))
        term2 = -0.5*(x - mu)@np.linalg.pinv(cov)@(x-mu).T
        return term1+term2
    
    
    stNormal = normal1.as_torch_distribution().as_normal()
    lp_torch = torch.sum(stNormal.log_prob(sample))
    lp_mine = torch.sum(normal1(sample))
    
    sample_np = sample.detach().cpu().numpy()
    lp_np = gaussian_log_prob(sample_np,std=normal1.scale.item())
    print("Log probability of sample - mine:{}, torch: {}, numpy: {}".format(lp_mine.item(), lp_torch.item(),lp_np))
    
    # Test if KL divergences match
    tNormal = torch_Normal(torch.randn(*shape),torch.ones(shape))
    kldiv_my = kl_normal_standardnormalprior(tNormal,normal1.as_torch_distribution())
    kldiv_torch = kl_divergence(tNormal,stNormal)
    print("KL divergence with a Normal - expecting {}, calculated {}".format(kldiv_torch,kldiv_my))
    print("KL divergence sums: expecting {}, calculated {}".format(torch.sum(kldiv_torch),torch.sum(kldiv_my)))
    
    # Test if my KL is zero for identical:
    kldiv_my = kl_normal_standardnormalprior(stNormal,normal1.as_torch_distribution())
    print("KL divergence with self - expecting {}, calculated {}".format(torch.zeros_like(kldiv_my),kldiv_my))
    
    # Check if we have approximately unit std output when using for matrix multiplications:
    input = torch.randn(1000,100)
    normal1 = StandardNormalPrior([1,100])
    normal1.standard_output_init()
    samples = normal1.sample(1000)
    out = torch.bmm(input.unsqueeze(1),samples.transpose(1,2)).detach().cpu().numpy()
    print("Std of standard output init samples multiplied with normal vectors: {}".format(np.std(out)))
    
    

# Unit testing:
if __name__ == '__main__':
    unit_test()
    
    
        












