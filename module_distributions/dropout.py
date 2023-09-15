import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number
from torch.nn import init
import numpy as np


class DropoutDistribution(Distribution):
    
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        self.num_elements = np.prod(self.loc.shape)
        super(DropoutDistribution, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RadialDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(RadialDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        if len(sample_shape) == 0:
            r_shape = [1]*len(shape)
            eps_shape = [1,self.num_elements]
        else:
            r_shape = [shape[0]] + [1]*(len(shape)-1)
            eps_shape = [shape[0],self.num_elements]
        # Normalized noise into unit ball and then multiply with "r" which is the new radius
        mu = self.loc.expand(shape)
        std = self.scale.expand(shape)
        epsilon_ = torch.randn_like(std)
        # Normalize epsilon to unit ball:
        epsilon_ = epsilon_.view(*eps_shape)
        epsilon = F.normalize(epsilon_)
        epsilon = epsilon.view(shape)
        r = torch.randn(*r_shape).type_as(mu)
        return mu + std * epsilon * r

    def log_prob(self,input):
        # The Appendix equation 44 suggests that the log probability
        # of a sample is proportional to -0.5 * ||(x - mean) / std||^2 
        # It is assumed that the shape of input is (*,mu.shape[0],mu.shape[1],..,mu.shape[-1])
        sum_axis = tuple(np.arange(-1,-len(self.batch_shape)-1,-1))
        mu = self.loc.expand_as(input)
        std = self.scale.expand_as(input)
        diff = input-mu
        scaled = diff / std
        lpdfs = -0.5*torch.sum(scaled.pow(2),dim=sum_axis)
        return lpdfs

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        # This is true up to a constant as described in the paper
        # The entropy is incorrect in the paper, it is missing a negative sign.
        # That is fixed here for clarity
        return torch.sum(torch.log(self.scale))
    
    def L_entropy(self):
        # This is for the \mathcal{L}_{entropy} term in the paper (not entropy)
        return -torch.log(self.scale)
        
    def L_entropy_total(self):
        # This is the \mathcal{L}_{entropy} term in the paper (not entropy)
        return torch.sum(self.L_entropy())

class Radial(nn.Module):
    def __init__(self,
                 shape,
                 loc_requires_grad=True,
                 scale_requires_grad=True,
                 divergence_estimation_samples=10):
        super(Radial,self).__init__()
        """
        divergence_estimation_samples controls the number of samples used to compute
        for example the KL divergence MC estimator.
        """
        if not hasattr(shape,'__len__'):
            shape = [shape]
        self.shape = shape
        self.loc = Parameter(torch.Tensor(*shape),requires_grad=loc_requires_grad)
        self.softplus_scale = Parameter(torch.Tensor(*shape),requires_grad=scale_requires_grad)
        self.divergence_estimation_samples = divergence_estimation_samples
        # Init parameters as standard normal by default:
        self.standard_init()
        
    def standard_init(self):
        # Standard normal initialization
        init.constant_(self.loc,0.0)
        # softplus(softplus_std) = std -> softplus(std) = softplus_std
        init.constant_(self.softplus_scale,np.log(np.exp(1.0)-1.0))
        
    def constant_init(self,loc_value=None,scale_value=None,softplus_scale_value=None):
        if loc_value is None and scale_value is None and softplus_scale_value is None:
            raise ValueError("all initialization values to constant init are None!")
        if loc_value is not None:
            init.constant_(self.loc,loc_value)
        if scale_value is not None:
            if not torch.is_tensor(scale_value):
                scale_value = torch.from_numpy(scale_value)
            with torch.no_grad():
                softplus_scale_value = torch.log(torch.exp(scale_value)-1.0)
            init.constant_(self.softplus_scale,softplus_scale_value)
        if softplus_scale_value is not None:
            init.constant_(self.softplus_scale,softplus_scale_value)
    
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
        init.constant_(self.loc,0.0)
        fan_in = 1 if len(self.shape) == 1 else np.prod(self.shape[1:])
        init.constant_(self.softplus_scale,np.log(np.exp(1.0/np.sqrt(fan_in))-1.0))
    
    def VI_init(self):
        if self.loc.ndim==1:
            # Assume that this is bias:
            init.constant_(self.loc,0.0)
            init.constant_(self.softplus_scale,1.0)
        else:
            fan_in = np.prod(self.shape[1:])
            std = 1/np.sqrt(fan_in)
            init.normal_(self.loc,std=std)
            softplus_std = np.log(np.exp(std)-1.0)
            init.constant_(self.softplus_scale,softplus_std)
    
    def orthogonal_init(self):
        # Initialize in orthogonal manner
        # If dim is 1 then there is no orthogonality:
        if self.loc.ndim==1:
            # Assume that this is bias:
            init.constant_(self.loc,0.0)
            init.constant_(self.softplus_scale,1.0)
        else:
            init.orthogonal_(self.loc)
            fan_in = np.prod(self.shape[1:])
            std = 1/np.sqrt(fan_in)
            softplus_std = np.log(np.exp(std)-1.0)
            init.constant_(self.softplus_scale,softplus_std)
    
    def custom_init(self,function):
        function(self)
    
    def freeze_scale(self):
        self.softplus_scale.requires_grad = False
    
    def unfreeze_scale(self):
        self.softplus_scale.requires_grad = True
    
    def freeze_loc(self):
        self.loc.requires_grad = False
    
    def unfreeze_loc(self):
        self.loc.requires_grad = True
    
    def get_parameters(self):
        return self.loc,F.softplus(self.softplus_scale)
    
    def get_named_parameters(self):
        return {"loc": self.loc, "scale": F.softplus(self.softplus_scale)}
    
    def as_torch_distribution(self):
        loc,scale = self.get_parameters()
        dist = RadialDistribution(loc=loc,scale=scale)
        dist.divergence_estimation_samples = self.divergence_estimation_samples
        return dist
    
    def forward(self,x):
        # Evaluates the log-pdf of the sample x
        loc,scale = self.get_parameters()
        return RadialDistribution(loc=loc,scale=scale).log_prob(x)
    
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
    
    
def kl_radial_normal(p, q):
    # This function uses the same format as the Normal KLD in pytorch,
    # i.e. the KLD is computed for each element individually and not reduced
    # Using notation in the paper https://arxiv.org/pdf/1907.00865.pdf
    L_entropy = p.L_entropy()
    samples = p.rsample([p.divergence_estimation_samples])
    L_cross_entropy = torch.mean(q.log_prob(samples),0) # Average over the MC samples
    return L_entropy - L_cross_entropy    

def unit_test():
    shape = [1,5]
    n_samples = 3
    radial_dist = Radial(shape)
    sample = radial_dist.sample(n_samples)
    print("Sample shape standard radial - correct: {}, result: {}".format(torch.Size([n_samples] + shape), sample.shape))
    lp_torch = torch.sum(radial_dist(sample))
    
    def radial_log_prob(x,mu=0.0,std=1.0):
        x = x.flatten()
        if not hasattr(mu,'__len__'):
            mu = np.zeros_like(x)+mu
        if not hasattr(std,'__len__'):
            std = np.zeros_like(x)+std
        return -0.5*np.sum(((x-mu)/std)**2)
    
    sample_np = sample.detach().cpu().numpy()
    lp_np = radial_log_prob(sample_np)
    print("Log probability of sample - torch: {}, numpy: {}".format(lp_torch.item(),lp_np))
    
    
    import torch.distributions as td
    kldiv = kl_radial_normal(radial_dist.as_torch_distribution(),td.Normal(torch.zeros(shape),torch.ones(shape)))
    print("KL divergence with standard normal {}".format(kldiv))
    
    radial_dist2 = Radial(shape)
    radial_dist2.VI_init()
    
    kldiv_torch = kl_radial_normal(radial_dist2.as_torch_distribution(),td.Normal(torch.zeros(shape),torch.ones(shape)))
    
    print("KL divergence with VI init || standard normal: {}".format(kldiv_torch))
    
    # In 1D the radial should match a normal distribution, so verify this!
    radial_dist = Radial([1])
    samples = radial_dist.sample(10000).detach().cpu().numpy()
    r = [np.min(samples),np.max(samples)]
    space = np.linspace(r[0],r[1],1000)
    pdf = td.Normal(torch.zeros([1]),torch.ones([1])).log_prob(torch.from_numpy(space)).exp().detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.hist(samples,density=True,bins=100)
    plt.plot(space,pdf)
    plt.show()
    

# Unit testing:
if __name__ == '__main__':
    unit_test()
    
    
        











