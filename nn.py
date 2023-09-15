# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, List, Callable
import torch.distributions as td
from functools import partial
from torch.autograd import Function
import warnings
# module_distributions contains parameterized distributions
import module_distributions as md

"""
The layers provided in this file support two ways of doing forward.
    1) the simplest way is to use these layers as
       if you were using pure pytorch. This means that one sample is used
       for every input.
    2) give a list with a length of N to the layer, and it will sample a unique
       parameter for every element. This means that you can repeat your input for example
       5 times to a list "input = [tensor]*5; output = network(input)" and you have
       5 samples for each input that are held in a list.

MC sampling is performed as a simple loop.

There are also methods of reducing the gradient estimator variance.
flipout is useful for convolutional layers and local reparametrization trick for
fully-connected ones.

If you use the "naive" gradient variance estimator, then a unique parameter sample is used
for every input, i.e. every batch element given + every sample in a list. This is VERY slow!
"""

""" UTILITIES """
class MCSampler(nn.Module):
    def __init__(self,
                 model,
                 mc_samples = None,
                 stack_result = True):
        super(MCSampler, self).__init__()
        self.mc_samples = mc_samples
        self.model = model
        self.stack_result = stack_result
        
    def forward(self,x,mc_samples=None):
        if mc_samples is None:
            mc_samples = self.mc_samples
        if mc_samples is None:
            raise ValueError("mc_samples was not given in init or forward!")
        output = self.model([x]*mc_samples)
        if self.stack_result:
            output = torch.stack(output,dim=0)
        return output


class Wrap(nn.Module):
    """
    Used to wrap any pytorch layer such that it can be applied to lists.
    Useful for activation functions with many mc samples for example.
    
    Also useful if you convert an abritrary neural network to BNN, because the
    conversion scripts in utils cannot convert custom layers (because of recursions and skip connections etc.)
    """
    def __init__(self,pytorch_module):
        super(Wrap, self).__init__()
        self.pytorch_module = pytorch_module
        
    def forward(self,x):
        if not isinstance(x,list):
            return self.pytorch_module(x)
        else:
            return [self.pytorch_module(sample) for sample in x]




""" PARENT MODULE FOR VARIATIONAL LAYERS """
class VariationalLayer(nn.Module):
    def __init__(self):
        super(VariationalLayer,self).__init__()
        # Use moduledicts so that the parameters of the distribution modules
        # are correctly registered.
        # MC samples are handled outside by feeding copies of the input to this
        # as a list! If only single is used then you can give the same input as to a normal
        # pytorch nn module.
        self.variational_posteriors = nn.ModuleDict()
        self.priors = nn.ModuleDict()
        
    def freeze_variational_posterior(self):
        for k,v in self.variational_posteriors.items():
            v.requires_grad_(False)
            
    def freeze_prior(self):
        for k,v in self.priors.items():
            v.requires_grad_(False)
            
    def unfreeze_variational_posterior(self):
        for k,v in self.variational_posteriors.items():
            v.requires_grad_(True)
            
    def unfreeze_prior(self):
        for k,v in self.priors.items():
            v.requires_grad_(True)
            
    def freeze(self):
        self.freeze_variational_posterior()
        self.freeze_prior()
        
    def unfreeze(self):
        self.unfreeze_variational_posterior()
        self.unfreeze_prior()
    
    def variational_posterior_init(self,method,keys=None):
        if keys is None:
            keys = list(self.variational_posteriors.keys())
        elif not hasattr(keys,"__len__"):
            keys = [keys]
        for key in keys:
            if isinstance(method,str):
                getattr(self.variational_posteriors[key],method)()
            else:
                method(self.variational_posteriors[key])
            
    def prior_init(self,method,keys=None):
        if keys is None:
            keys = list(self.priors.keys())
        elif not hasattr(keys,"__len__"):
            keys = [keys]
        for key in keys:
            if isinstance(method,str):
                getattr(self.priors[key],method)()
            else:
                method(self.priors[key])
            


class VILinear(VariationalLayer):
    def __init__(self, 
                 in_features = None,
                 out_features = None,
                 bias = True,
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        super(VILinear, self).__init__()
        if (in_features is None or out_features is None) and pytorch_module is None:
            raise ValueError("You need to either supply an existing pytorch nn.Module or in_features and out_features!")
        if pytorch_module is not None:
            pytorch_module = pytorch_module
        else:
            pytorch_module = nn.Linear(in_features = in_features,
                                       out_features = out_features,
                                       bias = bias,
                                       device = device,
                                       dtype = dtype)
        self.in_features = in_features
        self.out_features = out_features
        
        self.variational_posteriors.update({"weight":weight_posterior(pytorch_module.weight.shape)})
        self.priors.update({"weight":weight_prior(pytorch_module.weight.shape)})
        self.bias = bias
        if self.bias:
            self.variational_posteriors.update({"bias":bias_posterior(pytorch_module.bias.shape)})
            self.priors.update({"bias":bias_prior(pytorch_module.bias.shape)})
        
        # Initialize posterior
        self.reset_parameters()
        self.train_prior = train_prior
        if not train_prior:
            self.freeze_prior()
        # Put to device if given
        if device is not None:
            self.to(device)
            
        # Record the gradient variance reduction method and see if it is suitable:
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "lrt" and weight_posterior != md.Normal:
            raise ValueError("Cannot use local reparametrization trick (lrt) if posterior is not Normal!")
        if not (gradient_variance_reduction_method in ["none","lrt","bmm"]):
            raise ValueError("gradient_variance_reduction_method {} not understood!\
                             Needs to be in [None, lrt, bmm] \
                             which correspond to none, local reparametrization trick,\
                             and batch matrix multiplication".format(gradient_variance_reduction_method))
        
        self.gradient_variance_reduction_method = gradient_variance_reduction_method

    def reset_parameters(self) -> None:
        # The default parameterization is uniform kaiming like init for posterior.
        # Priors are always initialized by the module to some standard.
        self.variational_posteriors["weight"].VI_init()
        if "bias" in self.variational_posteriors:
            self.variational_posteriors["bias"].VI_init()

    def forward(self, input: Union[Tensor,List]):
        if isinstance(input,list):
            return_as_list = True
        else:
            input = [input]
            return_as_list = False
        if self.gradient_variance_reduction_method.lower() == "none":
            output = self.no_gradient_variance_reduction_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "lrt":
            output = self.local_reparametrization_trick(input)
        elif self.gradient_variance_reduction_method.lower() == "bmm":
            output = self.bmm_gradient_reduction_method(input)
        return output if return_as_list else output[0] # There is only one element in the list in the latter case
    
    def no_gradient_variance_reduction_forward(self,input):
        """
        This is the simplest forward method which uses one MC sample across the batch dimension
        """
        MC_samples = len(input)
        weights = self.variational_posteriors["weight"].sample(MC_samples)
        if self.bias:
            biases = self.variational_posteriors["bias"].sample(MC_samples)
        else:
            biases = [None]*MC_samples
        outputs = []
        for input_,weight,bias in zip(input,weights,biases):
            outputs.append(F.linear(input_, weight, bias))
        return outputs
    
    def local_reparametrization_trick(self,input):
        """
        Local reparametrization trick: https://arxiv.org/abs/1506.02557
        This uses basically the linearly transformed Gaussian
        to directly sample the outputs.
        
        WARNING: Current implementation only for Gaussian (uncorrelated n-dimensional)
        
        For a single "j":th row of the weight matrix W, the distribution is 
        w ~ N(m_j,diag(Sigma_j)) -> w = m_j + e_j, where e_j ~ N(0,diag(Sigma_j))
        Let h = wx so that h_j = w_j x
        Then:
            E[h_j] = E[(m_j + e_j)x] = E[m_jx + e_jx] = m_jx
            Cov[h_j] = E[((m_j + e_j)x - m_jx)^T((m_j + e_j)x - m_jx)]
                     = E[(e_jx)^T(e_jx)]
                     = E[x^Te_j^Te_jx]
                     = x^Tdiag(Sigma_j).**2x   (.** elementwise pow)
                     = (x.*Sigma_j)^Tx
                     = sum_i x_i**2*Sigma_j_i
                     = (x.**2)^T Sigma_j
        
        Since the covariance is zero between every element i and j, (i!=j), the Cov[h_i,h_j] is zero (by the second last row)
        This means that when computed as E[((m_j + e_j)x - m_jx)^T((m_i + e_i)x - m_ix)] = E[(e_jx)^T(e_ix)]
        the covariance matrix is 0 and thus it suffices to only compute elementwise operations here
        
        This then enables to sample directly from the distribution of h with the reparametrization trick:
            h_j = E[h_j] + epsilon*sqrt(Cov[h_j]), epsilon ~ N(0,1)
        
        We can actually go a step further when adding the bias.
        if h_1 ~ N(m_1,S_1) and h_2 ~ N(m_2,S_2), epsilon_j ~ N(0,I)
        h_1 = m_1 + chol(S_1)epsilon_1, h_2 = m_2 + chol(S_2)epsilon_2
        E[h_1 + h_2] = m_1 + m_2
        Cov[h_1+h_2] = E[(chol(S_1)epsilon_1 + chol(S_2)epsilon_2)(chol(S_1)epsilon_1 + chol(S_2)epsilon_2)^T]
                     = S_1 + S_2 (as cross terms have uncorrelated epsilons)
        """
        mu_weight,sigma_weight = self.variational_posteriors["weight"].get_parameters()
        if self.bias:
            mu_bias,sigma_bias = self.variational_posteriors["bias"].get_parameters()
            if input[0].ndim == 2:
                mu_bias = mu_bias.unsqueeze(0)
                sigma_bias = sigma_bias.unsqueeze(0)
            elif input[0].ndim == 3:
                # Unsqueeze twice
                mu_bias = mu_bias.unsqueeze(0)
                sigma_bias = sigma_bias.unsqueeze(0)
                mu_bias = mu_bias.unsqueeze(0)
                sigma_bias = sigma_bias.unsqueeze(0)
                
        MC_samples = len(input)
        """
        Concat along the first dimension. It does not affect the calculations.
        """
        input_ = torch.cat(input,dim=0)
        h_expectation = F.linear(input_,mu_weight)
        h_variance = F.linear(input_.pow(2),sigma_weight.pow(2))
        if self.bias:
            h_expectation = h_expectation + mu_bias
            h_variance = h_variance + sigma_bias.pow(2)
        
        h = h_expectation + torch.randn_like(h_variance)*torch.sqrt(h_variance) # Reparametrization trick
        """
        Chunk to the list of MC samples representation
        """
        outputs = list(torch.chunk(h,MC_samples,0))
        return outputs
    
    def bmm_gradient_reduction_method(self,input):
        if input[0].ndim == 3:
            raise ValueError("bmm gradient variance reduction only implemented for 1D inputs (2D with batch) \
                             This means that format Sequence-length,batch-size,dimension is not supported!")
        # This can be done in parallel across the samples
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        weights = self.variational_posteriors["weight"].sample(MC_samples*batch_size).permute(0,2,1)
        input = torch.cat(input,dim=0).unsqueeze(1)
        output = torch.bmm(input,weights).squeeze(1)
        if self.bias:
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            output = output + biases
        outputs = list(torch.chunk(output,MC_samples,0))
        return outputs

    def extra_repr(self):
        s_prob = "weight_posterior={}".format(type(self.variational_posteriors["weight"]).__name__)
        s_prob += ", weight_prior={}".format(type(self.priors["weight"]).__name__)
        if self.bias:
            s_prob += ", bias_posterior={}".format(type(self.variational_posteriors["bias"]).__name__)
            s_prob += ", bias_prior={}".format(type(self.priors["weight"]).__name__)
        s_prob += ", train_prior={}".format(self.train_prior)
        s = 'in_features={}, out_features={}, bias={}, '.format(self.in_features, self.out_features, self.bias is not None)
        return s + s_prob


""" PARENT MODULE FOR VARIATIONAL CONVOLUTIONAL LAYERS """
class VariationalConvolutionalLayer(VariationalLayer):
    def __init__(self,
                 pytorch_module: nn.Module,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior: bool = False):
        super(VariationalConvolutionalLayer, self).__init__()
        """
        Here we will exploit the fact that pytorch ConvNd layers store all the relevant parameters
        and hyperparameters, AND that they have the method _conv_forward, which can accept any weights
        and biases. This is benefitial for the standard forward.
        """
        self.variational_posteriors.update({"weight":weight_posterior(pytorch_module.weight.shape)})
        self.priors.update({"weight":weight_prior(pytorch_module.weight.shape)})
        if pytorch_module.bias is not None:
            self.variational_posteriors.update({"bias":bias_posterior(pytorch_module.bias.shape)})
            self.priors.update({"bias":bias_prior(pytorch_module.bias.shape)})
            self.bias = True
        else:
            self.bias = None
        
        # Then remove these parameters:
        pytorch_module = pytorch_module.to("cpu") # Not sure if needed
        del pytorch_module.weight, pytorch_module.bias
        self.pytorch_module = pytorch_module
        self.pytorch_module.bias = self.bias
        
        self.reset_parameters()
        
        # Freeze the prior unless you want to train it.
        self.train_prior = train_prior
        if not train_prior:
            self.freeze_prior()

    def reset_parameters(self) -> None:
        # The default parameterization is uniform kaiming like init for posterior.
        # Priors are always initialized by the modules to some standard.
        self.variational_posteriors["weight"].VI_init()
        if "bias" in self.variational_posteriors:
            self.variational_posteriors["bias"].VI_init()

    def extra_repr(self):
        s_prob = ", weight_posterior={}".format(type(self.variational_posteriors["weight"]).__name__)
        s_prob += ", weight_prior={}".format(type(self.priors["weight"]).__name__)
        if self.bias:
            s_prob += ", bias_posterior={}".format(type(self.variational_posteriors["bias"]).__name__)
            s_prob += ", bias_prior={}".format(type(self.priors["weight"]).__name__)
        s_prob += ", train_prior={}".format(self.train_prior)
        s = self.pytorch_module.extra_repr()
        return s + s_prob
    
    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input,list):
            return_as_list = True
        else:
            input = [input]
            return_as_list = False
        if self.gradient_variance_reduction_method.lower() == "none":
            output = self.no_gradient_variance_reduction_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "flipout":
            output = self.flipout_gradient_reduction_conv_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "naive":
            output = self.naive_gradient_reduction_conv_forward(input)
        else:
            raise ValueError("Gradient variance reduction method {} not understood".format(self.gradient_variance_reduction_method))
        return output if return_as_list else output[0] # There is only one element in the list in the latter case
    
    def no_gradient_variance_reduction_forward(self,input):
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        dims = input[0].ndim
        weights = self.variational_posteriors["weight"].sample(MC_samples)
        outputs = [self.pytorch_module._conv_forward(input_,weight_,None) for input_,weight_ in zip(input,weights)]
        # We can add the unbiased bias like this:
        if self.bias:
            outputs = torch.cat(outputs,0)
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
            outputs = list(torch.chunk(outputs,MC_samples,0))
        return outputs
    
    def naive_gradient_reduction_conv_forward(self, input: Tensor):
        # Naive approach. Every input has unique sample with which they are convolved.
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        dims = input[0].ndim
        input = torch.cat(input,dim=0)
        weights = self.variational_posteriors["weight"].sample(MC_samples*batch_size)
        outputs = [self.pytorch_module._conv_forward(input_.unsqueeze(0),weight_,None) for input_,weight_ in zip(input,weights)]
        outputs = torch.cat(outputs,dim=0)
        if self.bias:
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
        output = list(torch.chunk(outputs,MC_samples,0))
        return output
    
    def flipout_gradient_reduction_conv_forward(self, input: Tensor):
        # FLIPOUT: EFFICIENT PSEUDO-INDEPENDENT WEIGHT PERTURBATIONS ON MINI-BATCHES
        # https://arxiv.org/pdf/1803.04386.pdf
        """
        Flipout works when:
            a) weights are considered to be independent
            b) distribution is symmetric around zero
        
        When a random variable can be decomposed as loc + pertubation*scale
        where the loc is deterministic and the pertubation*scale is zero mean and symmetric
        we should thus be able to use this one!
        
        (\hat(W) \odot r s^T)^T x = \hat(W)^T(x \odot s) \odot r
        In convolutions, this is typically not done via \hat(W) = toeplitz(W), but rather
        such that only the channels are modulated. See for example 
        https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
        """
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        # Infer pertubation shapes:
        dims = input[0].ndim
        spatial_singletons = tuple((1 for i in range(dims-2)))
        s_shape = (batch_size,self.pytorch_module.in_channels) + spatial_singletons
        r_shape = (batch_size,self.pytorch_module.out_channels) + spatial_singletons
        weights = self.variational_posteriors["weight"].sample(MC_samples)
        mu,sigma = self.variational_posteriors["weight"].get_parameters()
        zero_mean_pertubations = weights - mu.unsqueeze(0)
        expectations = [self.pytorch_module._conv_forward(input_,mu,None) for input_, in input] 
        outputs = []
        for input_,delta_w,expect_ in zip(input,zero_mean_pertubations,expectations):
            s = 2*torch.randint(size=s_shape,low=0,high=2,device=sigma.device,dtype=sigma.dtype) - 1
            xs = s*input_
            delta_wxs = self.pytorch_module._conv_forward(xs,delta_w,None)
            r = 2*torch.randint(size=r_shape,low=0,high=2,device=sigma.device,dtype=sigma.dtype) - 1
            delta_wxsr = delta_wxs*r
            outputs.append(expect_ + delta_wxsr)
        if self.bias:
            outputs = torch.cat(outputs,dim=0)
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
            outputs = list(torch.chunk(outputs,MC_samples,0))
        return outputs

class VIConv1d(VariationalConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride= 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.Conv1d layer!")
        if pytorch_module is None:
            pytorch_module = nn.Conv1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias,
                                       padding_mode=padding_mode,
                                       device=device,
                                       dtype=dtype)
        super(VIConv1d,self).__init__(pytorch_module,
                                      weight_posterior,
                                      weight_prior,
                                      bias_posterior,
                                      bias_prior,
                                      train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method
        
class VIConv2d(VariationalConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride= 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.Conv2d layer!")
        if pytorch_module is None:
            pytorch_module = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias,
                                       padding_mode=padding_mode,
                                       device=device,
                                       dtype=dtype)
        super(VIConv2d,self).__init__(pytorch_module,
                                      weight_posterior,
                                      weight_prior,
                                      bias_posterior,
                                      bias_prior,
                                      train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method
    
    
class VIConv3d(VariationalConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride= 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.Conv3d layer!")
        if pytorch_module is None:
            pytorch_module = nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias,
                                       padding_mode=padding_mode,
                                       device=device,
                                       dtype=dtype)
        super(VIConv3d,self).__init__(pytorch_module,
                                      weight_posterior,
                                      weight_prior,
                                      bias_posterior,
                                      bias_prior,
                                      train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method


""" PARENT MODULE FOR VARIATIONAL TRANSPOSE CONVOLUTIONAL LAYERS """
class VariationalTransposeConvolutionalLayer(VariationalLayer):
    def __init__(self,
                 pytorch_module: nn.Module,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior: bool = False):
        super(VariationalConvolutionalLayer, self).__init__()
        """
        Unfortunately we cannot exploit the "conv_forward" anymore, as transpose
        convolutional layers utilize the forward method directly while referencing
        to their own parameters. We will create a similar method to this class.
        """
        self.variational_posteriors.update({"weight":weight_posterior(pytorch_module.weight.shape)})
        self.priors.update({"weight":weight_prior(pytorch_module.weight.shape)})
        if pytorch_module.bias is not None:
            self.variational_posteriors.update({"bias":bias_posterior(pytorch_module.bias.shape)})
            self.priors.update({"bias":bias_prior(pytorch_module.bias.shape)})
            self.bias = True
        else:
            self.bias = None
        
        # Add the convolution function to properties
        if isinstance(pytorch_module,nn.ConvTranspose1d):
            self.convolution_function = F.conv_transpose1d
        elif isinstance(pytorch_module,nn.ConvTranspose2d):
            self.convolution_function = F.conv_transpose2d
        elif isinstance(pytorch_module,nn.ConvTranspose3d):
            self.convolution_function = F.conv_transpose3d
        else:
            raise ValueError("Cannot use pytorch module {} for transpose convolution!".format(type(pytorch_module).__name__))
        
        # Then remove these parameters:
        pytorch_module = pytorch_module.to("cpu") # Not sure if needed
        del pytorch_module.weight, pytorch_module.bias
        self.pytorch_module = pytorch_module
        self.pytorch_module.bias = self.bias
        
        self.reset_parameters()
        
        # Freeze the prior unless you want to train it.
        self.train_prior = train_prior
        if not train_prior:
            self.freeze_prior()

    def reset_parameters(self) -> None:
        # The default parameterization is uniform kaiming like init for posterior.
        # Priors are always initialized by the modules to some standard.
        self.variational_posteriors["weight"].VI_init()
        if "bias" in self.variational_posteriors:
            self.variational_posteriors["bias"].VI_init()

    def extra_repr(self):
        s_prob = ", weight_posterior={}".format(type(self.variational_posteriors["weight"]).__name__)
        s_prob += ", weight_prior={}".format(type(self.priors["weight"]).__name__)
        if self.bias:
            s_prob += ", bias_posterior={}".format(type(self.variational_posteriors["bias"]).__name__)
            s_prob += ", bias_prior={}".format(type(self.priors["weight"]).__name__)
        s_prob += ", train_prior={}".format(self.train_prior)
        s = self.pytorch_module.extra_repr()
        return s + s_prob
    
    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input,list):
            return_as_list = True
        else:
            input = [input]
            return_as_list = False
        if self.gradient_variance_reduction_method.lower() == "none":
            output = self.no_gradient_variance_reduction_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "flipout":
            output = self.flipout_gradient_reduction_conv_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "naive":
            output = self.naive_gradient_reduction_conv_forward(input)
        else:
            raise ValueError("Gradient variance reduction method {} not understood".format(self.gradient_variance_reduction_method))
        return output if return_as_list else output[0] # There is only one element in the list in the latter case
    
    def compute_convolution(self,input,weight,bias=None,output_size=None):
        # Compute output padding via the pytorch_module:
        output_padding = self.pytorch_module._output_padding(input,
                                                             output_size,
                                                             self.pytorch_module.stride,
                                                             self.pytorch_module.padding,
                                                             self.pytorch_module.kernel_size,
                                                             self.pytorch_module.dilation)
        # Compute the convolution:
        return self.convolution_function(input, 
                                         weight, 
                                         bias, 
                                         self.pytorch_module.stride, 
                                         self.pytorch_module.padding,
                                         output_padding, 
                                         self.pytorch_module.groups, 
                                         self.pytorch_module.dilation)
    
    def no_gradient_variance_reduction_forward(self,input):
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        dims = input[0].ndim
        weights = self.variational_posteriors["weight"].sample(MC_samples)
        outputs = [self.compute_convolution(input_,weight_,None) for input_,weight_ in zip(input,weights)]
        # We can add the unbiased bias like this:
        if self.bias:
            outputs = torch.cat(outputs,0)
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
            outputs = list(torch.chunk(outputs,MC_samples,0))
        return outputs
    
    def naive_gradient_reduction_conv_forward(self, input: Tensor):
        # Naive approach. Every input has unique sample with which they are convolved.
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        dims = input[0].ndim
        input = torch.cat(input,dim=0)
        weights = self.variational_posteriors["weight"].sample(MC_samples*batch_size)
        outputs = [self.compute_convolution(input_.unsqueeze(0),weight_,None) for input_,weight_ in zip(input,weights)]
        outputs = torch.cat(outputs,dim=0)
        if self.bias:
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
        output = list(torch.chunk(outputs,MC_samples,0))
        return output
    
    def flipout_gradient_reduction_conv_forward(self, input: Tensor):
        # FLIPOUT: EFFICIENT PSEUDO-INDEPENDENT WEIGHT PERTURBATIONS ON MINI-BATCHES
        # https://arxiv.org/pdf/1803.04386.pdf
        """
        Flipout works when:
            a) weights are considered to be independent
            b) distribution is symmetric around zero
        
        When a random variable can be decomposed as loc + pertubation*scale
        where the loc is deterministic and the pertubation*scale is zero mean and symmetric
        we should thus be able to use this one!
        
        (\hat(W) \odot r s^T)^T x = \hat(W)^T(x \odot s) \odot r
        In convolutions, this is typically not done via \hat(W) = toeplitz(W), but rather
        such that only the channels are modulated. See for example 
        https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
        """
        MC_samples = len(input)
        batch_size = input[0].shape[0]
        # Infer pertubation shapes:
        dims = input[0].ndim
        spatial_singletons = tuple((1 for i in range(dims-2)))
        s_shape = (batch_size,self.pytorch_module.in_channels) + spatial_singletons
        r_shape = (batch_size,self.pytorch_module.out_channels) + spatial_singletons
        weights = self.variational_posteriors["weight"].sample(MC_samples)
        mu,sigma = self.variational_posteriors["weight"].get_parameters()
        zero_mean_pertubations = weights - mu.unsqueeze(0)
        expectations = [self.compute_convolution(input_,mu,None) for input_, in input] 
        outputs = []
        for input_,delta_w,expect_ in zip(input,zero_mean_pertubations,expectations):
            s = 2*torch.randint(size=s_shape,low=0,high=2,device=sigma.device,dtype=sigma.dtype) - 1
            xs = s*input_
            delta_wxs = self.compute_convolution(xs,delta_w,None)
            r = 2*torch.randint(size=r_shape,low=0,high=2,device=sigma.device,dtype=sigma.dtype) - 1
            delta_wxsr = delta_wxs*r
            outputs.append(expect_ + delta_wxsr)
        if self.bias:
            outputs = torch.cat(outputs,dim=0)
            biases = self.variational_posteriors["bias"].sample(MC_samples*batch_size)
            for i in range(dims-2):
                biases = biases.unsqueeze(-1)
            outputs = outputs + biases
            outputs = list(torch.chunk(outputs,MC_samples,0))
        return outputs


class VIConvTranspose1d(VariationalTransposeConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 groups = 1,
                 bias = True,
                 dilation = 1,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.ConvTranspose1d layer!")
        if pytorch_module is None:
            pytorch_module = nn.ConvTranspose1d(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=kernel_size, 
                                                stride=stride, 
                                                padding=padding, 
                                                output_padding=output_padding, 
                                                groups=groups, 
                                                bias=bias, 
                                                dilation=dilation, 
                                                padding_mode=padding_mode, 
                                                device=device, 
                                                dtype=dtype)
        super(VIConvTranspose1d, self).__init__(pytorch_module,
                                                weight_posterior,
                                                weight_prior,
                                                bias_posterior,
                                                bias_prior,
                                                train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method


class VIConvTranspose2d(VariationalTransposeConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 groups = 1,
                 bias = True,
                 dilation = 1,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.ConvTranspose2d layer!")
        if pytorch_module is None:
            pytorch_module = nn.ConvTranspose2d(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=kernel_size, 
                                                stride=stride, 
                                                padding=padding, 
                                                output_padding=output_padding, 
                                                groups=groups, 
                                                bias=bias, 
                                                dilation=dilation, 
                                                padding_mode=padding_mode, 
                                                device=device, 
                                                dtype=dtype)
        super(VIConvTranspose2d, self).__init__(pytorch_module,
                                                weight_posterior,
                                                weight_prior,
                                                bias_posterior,
                                                bias_prior,
                                                train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method


class VIConvTranspose3d(VariationalTransposeConvolutionalLayer):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 kernel_size = None,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 groups = 1,
                 bias = True,
                 dilation = 1,
                 padding_mode = 'zeros',
                 device = None,
                 dtype = None,
                 pytorch_module = None,
                 weight_posterior: nn.Module = md.Normal,
                 weight_prior: nn.Module = md.StandardNormalPrior,
                 bias_posterior: nn.Module = md.Normal,
                 bias_prior: nn.Module = md.StandardNormalPrior,
                 train_prior = False,
                 gradient_variance_reduction_method = None):
        if (in_channels is None or out_channels is None or kernel_size is None)\
        and pytorch_module is None:
            raise ValueError("in_channels or out_channels or kernel_size was None while also pytorch_module being None.\
                             Either supply the convolution arguments or an already instantiated nn.ConvTranspose3d layer!")
        if pytorch_module is None:
            pytorch_module = nn.ConvTranspose3d(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=kernel_size, 
                                                stride=stride, 
                                                padding=padding, 
                                                output_padding=output_padding, 
                                                groups=groups, 
                                                bias=bias, 
                                                dilation=dilation, 
                                                padding_mode=padding_mode, 
                                                device=device, 
                                                dtype=dtype)
        super(VIConvTranspose3d, self).__init__(pytorch_module,
                                                weight_posterior,
                                                weight_prior,
                                                bias_posterior,
                                                bias_prior,
                                                train_prior)
        
        gradient_variance_reduction_method = str(gradient_variance_reduction_method).lower()
        if gradient_variance_reduction_method == "flipout" and weight_posterior not in [md.Normal,md.Radial]:
            raise ValueError("Cannot use flipout if posterior is not Normal or Radial!")
        if not (gradient_variance_reduction_method in ["none","flipout","naive"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, flipout, naive]".format(gradient_variance_reduction_method))
        self.gradient_variance_reduction_method = gradient_variance_reduction_method
    


class StraightThroughQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class VariationalDropoutLayer(nn.Module):
    r"""
    This layer treats p as the KEEP probability!
    
    Args
        pytorch_module: the module, such as nn.Linear or nn.ConvNd to be used
        
        dropout_initialization: initial dropout probability
        
        dropout_pattern: either "input" or "output", wether the dropout is applied on the input
        or on the output i.e. weight column pattern or row pattern.
        
        dropout_dim: which dimension of the input is considered the drop out dim. 1 is usually the channels/neurons
        
        dropout_distribution: the distribution from which the dropout mask is drawn. Take consideration that the gradients can be computed
        must be the class and not an instantiation of that class (e.g. td.Bernoulli and not td.Bernoulli(probs=something))
        
        independent_dropout: wether to use independent dropout probabilities for each dimension (not just one p but a vector),
        
        force_quantize: wether to quantize the output of the dropout_distribution, for example in the case of the relaxed
        bernoulli, clamp the mask to 0 or 1. Gradient is estimated by straight through estimator.
        
        weight_regularizer: the weight and the bias are assumed to be quantized Gaussian. This is the 
        reciprocal of the 2 * variance of the Gaussian, so that the log likelihood log p(w) \propto
        1/(2 var) w^Tw = weight_regularizer * w^Tw
    """
    def __init__(self,
                 pytorch_module: nn.Module,
                 dropout_initialization: float = 0.5,
                 dropout_pattern: Union[str,list] = "input",
                 dropout_dim: Union[int,list] = 1,
                 dropout_distribution: td.Distribution = partial(td.RelaxedBernoulli,temperature=0.1),
                 independent_dropout: bool = False,
                 force_quantize: bool = False,
                 weight_regularizer: float = 0.01,
                 gradient_variance_reduction_method: str = "batch_mask"
                 ):
        super().__init__()
        if not (gradient_variance_reduction_method in ["none","batch_mask"]):
            raise ValueError("gradient_variance_reduction_method {} not understood! Needs to be in [None, batch_mask]".format(gradient_variance_reduction_method))
        if dropout_pattern not in ["input", "output"]:
            raise ValueError("Cannot understand dropout_pattern: {}, Must be in [input,output]".format(dropout_pattern))
        self.pytorch_module = pytorch_module
        dropout_initialization = torch.ones((1,)) * dropout_initialization
        logit_init = -torch.log(1/dropout_initialization - 1) # Inverse of sigmoid
        self.input_shape = self.pytorch_module.weight.shape[1]
        self.output_shape = self.pytorch_module.weight.shape[0]
        if dropout_pattern == "input":
            if independent_dropout:
                self.input_logits = nn.Parameter((logit_init*torch.ones((self.input_shape,))).float())
            else:
                self.input_logits = nn.Parameter((logit_init*torch.ones((1,))).float())
        else:
            self.input_logits = None
        if dropout_pattern == "output":
            if independent_dropout:
                self.output_logits = nn.Parameter((logit_init*torch.ones((self.output_shape,))).float())
            else:
                self.output_logits = nn.Parameter((logit_init*torch.ones((1,))).float())
        else:
            self.output_logits = None
        
        self.dropout_pattern = dropout_pattern
        self.apply_input_dropout = dropout_pattern in ["input"]
        self.apply_output_dropout = dropout_pattern in ["output"]
        self.independent_dropout = independent_dropout
        self.force_quantize = force_quantize
        self.weight_regularizer = weight_regularizer
        self.dropout_distribution = dropout_distribution
        self.gradient_variance_reduction_method = gradient_variance_reduction_method
    
    def KL(self):
        # In the github repo for concrete dropout https://github.com/yaringal/ConcreteDropout
        # the KL divergence term is divided into two terms that are scaled independently
        # with two factors "weights_regularizer" and "dropout_regularizer"
        # This is actually a type of scaled KL divergence with a common scaler "alpha"
        # alpha * KL \propto alpha * (1-p)/(2*var)||w|| - alpha * H[p]
        # = (1-p) * weights_regularizer * ||w|| - alpha * H[p]
        # = (1-p) * weights_regularizer * ||w|| - dropout_regularizer * H[p]
        # So we can see that the weight_regularizer = dropout_regularizer / (2*var)
        # -> var = dropout_regularizer / (2 * weight_regularizer)
        # This means that the implied variance of the weights is modified by changing the 
        # dropout and or the weight regularizer.
        # This implementation will not use the "dropout_regularizer" because the KL term
        # can be scaled outside if one wishes and the weight_regularizer can then be modified
        # to account for this.
        
        # Note that the KL term is correct up to a constant. See Uncertainty in Deep Learning (p. 120, Yarin Gal)
        
        # Bias is only regularized with the output dropout pattern, as it is then drawn from the dropout distribution
        
        w,b = self.pytorch_module.weight,self.pytorch_module.bias
        if self.apply_input_dropout:
            logits = self.input_logits.expand(self.input_shape) # Expand has only effect when shape is 1
            sum_dims = (0,) + tuple(range(2,w.ndim))
            bias_norm = None
        elif self.apply_output_dropout:
            logits = self.output_logits.expand(self.output_shape) # Expand has only effect when shape is 1
            sum_dims = tuple(range(1,w.ndim))
            if b is not None:
                bias_norm = b.pow(2)
            else:
                bias_norm = None
        else:
            raise ValueError("Neither input_dropout or output_dropout has been specified!")
        
        weight_norm = w.pow(2).sum(sum_dims)
        p = torch.sigmoid(logits)
        negative_entropy = p*torch.log(p) + (1-p)*torch.log(1-p)
        # Constants are left out since the term that is proportional to the quantization level
        # will be -log(machine_epsilon) and thus dominate. Thus since we dont measure this constant
        # this will be either way not correct up to a constant
        weight_term = torch.sum(self.weight_regularizer * (1-p) * weight_norm + negative_entropy)
        if bias_norm is not None:
            bias_term = torch.sum(self.weight_regularizer * (1-p) * bias_norm + negative_entropy)
            return weight_term + bias_term
        else:
            return weight_term
        
    
    def extra_repr(self):
        if isinstance(self.dropout_distribution,partial):
            dropout_name = self.dropout_distribution.func.__name__
        else:
            dropout_name = self.dropout_distribution.__name__
        s_prob = "dropout_distribution={}".format(dropout_name)
        s_prob += ", dropout_pattern={}".format(self.dropout_pattern)
        s_prob += ", independent_dropout={}".format(self.independent_dropout)
        s_prob += ", force_quantize={}".format(self.force_quantize)
        s_prob += ", weight_regularizer={}".format(self.weight_regularizer)
        return s_prob
    
    def generate_dropout_mask(self,logits,batch_size):
        if not isinstance(batch_size,torch.Size):
            batch_size = torch.Size([batch_size])
        # Bernoulli is parameterized with p, which gives the success probability
        # our logits are for the drop probability. To turn them into success logits, we can use
        # p_drop = sigmoid(logits_drop) = 1 - sigmoid(logits_success)
        # 1 - sigmoid(logits_drop) = sigmoid(logits_success)
        # -log((1 - sigmoid(logits_drop))^-1 - 1) = logits_success
        # = -log((1 - sigmoid(logits_drop))^-1 - 1) = -logits_drop
        dist = self.dropout_distribution(logits=-logits)
        samples = dist.rsample(batch_size)
        if self.force_quantize:
            samples = StraightThroughQuantize.apply(samples)
        return samples
    
    def batch_mask_gradient_variance_reduction_forward(self,input):
        # This is in real life the only sensible way to do this as we get the best of MC dropout here
        MC_samples = len(input)
        x = torch.cat(input,dim=0)
        if self.apply_input_dropout:
            mask = self.generate_dropout_mask(self.input_logits,batch_size=x.shape[0])
            nd = x.ndim
            md = mask.ndim
            mask = mask.view(*mask.shape,*((1,)*(nd-md)))
            x = x*mask
        x = self.pytorch_module(x)
        if self.apply_output_dropout:
            mask = self.generate_dropout_mask(self.output_logits,batch_size=x.shape[0])
            nd = x.ndim
            md = mask.ndim
            mask = mask.view(*mask.shape,*((1,)*(nd-md)))
            x = x*mask
        return list(torch.chunk(x,MC_samples,0))
    
    def no_gradient_variance_reduction_forward(self,input):
        # This is mainly for curiosity and should not be used.
        # For example if you make visualizations of predictions on a grid, you should 
        # get more continuous contours.
        output = []
        for x in input:
            if self.apply_input_dropout:
                mask = self.generate_dropout_mask(self.input_logits,batch_size=1)
                nd = x.ndim
                md = mask.ndim
                mask = mask.view(*mask.shape,*((1,)*(nd-md)))
                x = x*mask
            x = self.pytorch_module(x)
            if self.apply_output_dropout:
                mask = self.generate_dropout_mask(self.output_logits,batch_size=1)
                nd = x.ndim
                md = mask.ndim
                mask = mask.view(*mask.shape,*((1,)*(nd-md)))
                x = x*mask
            output.append(x)
        return output
        
    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input,list):
            return_as_list = True
        else:
            input = [input]
            return_as_list = False
        if self.gradient_variance_reduction_method.lower() == "batch_mask":
            output = self.batch_mask_gradient_variance_reduction_forward(input)
        elif self.gradient_variance_reduction_method.lower() == "none":
            output = self.no_gradient_variance_reduction_forward(input)
        else:
            raise ValueError("Gradient variance reduction method {} not understood".format(self.gradient_variance_reduction_method))
        return output if return_as_list else output[0] # There is only one element in the list in the latter case
    
    







    
        
