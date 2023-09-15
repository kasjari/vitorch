import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions import Normal as torch_Normal
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np

class Normal(nn.Module):
    def __init__(self,
                 shape,
                 loc_requires_grad=True,
                 scale_requires_grad=True,
                 divergence_estimation_samples=10):
        super(Normal,self).__init__()
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
        dist = torch_Normal(loc=loc,scale=scale)
        dist.divergence_estimation_samples = self.divergence_estimation_samples
        return dist
    
    def forward(self,x):
        # Evaluates the log-pdf of the sample x
        loc,scale = self.get_parameters()
        return torch_Normal(loc=loc,scale=scale).log_prob(x)
    
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
    
    
    

def unit_test():
    shape = [1,5]
    n_samples = 3
    normal1 = Normal(shape)
    sample = normal1.sample(n_samples)
    print("Sample shape standard normal - correct: {}, result: {}".format(torch.Size([n_samples] + shape), sample.shape))
    lp_torch = torch.sum(normal1(sample))
    
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
    
    sample_np = sample.detach().cpu().numpy()
    lp_np = gaussian_log_prob(sample_np)
    print("Log probability of sample - torch: {}, numpy: {}".format(lp_torch.item(),lp_np))
    
    
    kldiv = kl_divergence(normal1.as_torch_distribution(),normal1.as_torch_distribution())
    print("KL divergence with self - expecting {}, calculated {}".format(torch.zeros(shape),kldiv))
    
    normal2 = Normal(shape)
    normal2.VI_init()
    
    def numpy_KL(p,q):
        p_mu,p_std = p.get_parameters()
        q_mu,q_std = q.get_parameters()
        
        p_mu = p_mu.detach().cpu().numpy().flatten()
        p_std = p_std.detach().cpu().numpy().flatten()
        q_mu = q_mu.detach().cpu().numpy().flatten()
        q_std = q_std.detach().cpu().numpy().flatten()
        
        # From wikipedia:
        p_cov = np.diag(p_std)**2
        q_cov = np.diag(q_std)**2
        
        t1 = np.trace(np.linalg.pinv(q_cov) @ p_cov)
        t2 = (q_mu - p_mu)@np.linalg.pinv(q_cov)@(q_mu - p_mu).T
        t3 = -np.float32(len(p_mu))
        t4 = np.log(np.linalg.det(q_cov)/np.linalg.det(p_cov))
        
        return 0.5*(t1+t2+t3+t4)
    
    kldiv_torch = torch.sum(kl_divergence(normal2.as_torch_distribution(),normal1.as_torch_distribution()))
    kldiv_numpy = numpy_KL(normal2,normal1)
    
    print("KL divergence random VI init || standard normal - torch: {}, numpy: {}".format(kldiv_torch.item(),kldiv_numpy.item()))
    
    
    normal2 = Normal(shape)
    normal2.orthogonal_init()
    kldiv_torch = torch.sum(kl_divergence(normal2.as_torch_distribution(),normal1.as_torch_distribution()))
    kldiv_numpy = numpy_KL(normal2,normal1)
    
    print("KL divergence random ortho init || standard normal - torch: {}, numpy: {}".format(kldiv_torch.item(),kldiv_numpy.item()))
    
    
    normal3 = Normal([2,3])
    print(normal3.sample().shape)
    
# Unit testing:
if __name__ == '__main__':
    unit_test()
    
    
        












