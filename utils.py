import nn as tpn
import module_distributions as md
import torch.nn as nn
import torch.distributions as td
from typing import Optional, Union, List, Callable
from functools import partial
import warnings

def replace_layers(network: nn.Module, # A neural network
                   replacement_rule: Callable, # A function which returns the approriate layer to be used
                   skip_layer: Union[List,str,nn.Module] = None):
    # Use skip layer to skip replacing some layers
    # can be a string for the full name of the layer, e.g. "model.encoder.conv1"
    # or it can be the layer it self, e.g. model.encoder.conv1
    # It can also be a list of them
    if not isinstance(skip_layer,list):
        skip_layer = [skip_layer]
    names_and_modules = list(network.named_modules())
    for name,module in names_and_modules:
        # Doing it this way will ensure that nn.Sequential objects will have their layers
        # correctly replaced.
        for subname,submodule in module._modules.items():
            full_name = name+"."+subname
            if full_name not in skip_layer and submodule not in skip_layer:
                module._modules[subname] = replacement_rule(submodule)
    return network


def variational_dropout_replacement_rule(module,
                                         dropout_arguments = {"dropout_initialization" : 0.5,
                                                              "dropout_pattern" : "input",
                                                              "dropout_dim" : 1,
                                                              "dropout_distribution" : partial(td.RelaxedBernoulli,temperature=0.1),
                                                              "independent_dropout" : False,
                                                              "force_quantize" : False,
                                                              "weight_regularizer" : 0.01}):
    """
    A ready made variational dropout replacement rule.
    Will not change layers that have 1 input dimension or 1 output dimension
    when those dimensions are specified to be dropped out 
    (you would get all zero output or all zero input sometimes)
    """
    if isinstance(module,nn.Linear) or isinstance(module,nn.modules.conv._ConvNd):
        if dropout_arguments["dropout_pattern"] == "input" and module.weight.shape[1] == 1:
            warnings.warn("Skipped a {} layer with 1 input channel/neuron and dropout specified to be applied to that dimension".format(module.__class__.__name__))
            return module # We do not want to drop 1 input channel
        elif dropout_arguments["dropout_pattern"] == "output" and module.weight.shape[0] == 1:
            warnings.warn("Skipped a {} layer with 1 output channel/neuron and dropout specified to be applied to that dimension".format(module.__class__.__name__))
            return module # We do not want to drop 1 output channel
        else:
            return tpn.VariationalDropoutLayer(module,**dropout_arguments)
    else:
        # This layer is not linear or conv so return itself
        return module


# Just a convenience function with default wrapping
def convert_to_dropout_network(network,
                               skip_layer = None,
                               dropout_initialization = 0.5,
                               dropout_pattern = "input",
                               dropout_dim = 1,
                               dropout_distribution = partial(td.RelaxedBernoulli,temperature=0.1),
                               independent_dropout = False,
                               force_quantize = False,
                               weight_regularizer = 0.01):
    if skip_layer is None:
        warnings.warn("\n\
    No skip_layer was provided to the function convert_to_dropout_network. \n\
    If you use dropout_pattern = input, then you might want to exclude the first layer. \n\
    Similarly, if you use dropout_pattern = output, then you might want to exclude the last layer. \n\
    This is to avoid situations where you dropout the data or the predictions. \n\
    This is just a warning and the program continues.")
    dropout_arguments = {"dropout_initialization" : dropout_initialization,
                         "dropout_pattern" : dropout_pattern,
                         "dropout_dim" : dropout_dim,
                         "dropout_distribution" : dropout_distribution,
                         "independent_dropout" : independent_dropout,
                         "force_quantize" : force_quantize,
                         "weight_regularizer" : weight_regularizer}
    replacement_rule = partial(variational_dropout_replacement_rule,dropout_arguments=dropout_arguments)
    return replace_layers(network,replacement_rule,skip_layer)


def MFVI_replacement_rule(module,
                          mfvi_arguments = {"weight_posterior" : md.Normal,
                                            "weight_prior" : md.StandardNormalPrior,
                                            "bias_posterior" : md.Normal,
                                            "bias_prior" : md.StandardNormalPrior,
                                            "train_prior" : False,
                                            "gradient_variance_reduction_method_linear" : None,
                                            "gradient_variance_reduction_method_convolution" : None}):
    if isinstance(module,nn.Linear):
        gvr = mfvi_arguments["gradient_variance_reduction_method_linear"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VILinear(pytorch_module=module,
                                  gradient_variance_reduction_method = gvr,
                                  **mydict)
    elif isinstance(module,nn.Conv1d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConv1d(pytorch_module=module,
                                  gradient_variance_reduction_method = gvr,
                                  **mydict)
    elif isinstance(module,nn.Conv2d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConv2d(pytorch_module=module,
                                  gradient_variance_reduction_method = gvr,
                                  **mydict)
    elif isinstance(module,nn.Conv3d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConv3d(pytorch_module=module,
                                  gradient_variance_reduction_method = gvr,
                                  **mydict)
    elif isinstance(module,nn.ConvTranspose1d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConvTranspose1d(pytorch_module=module,
                                           gradient_variance_reduction_method = gvr,
                                           **mydict)
    elif isinstance(module,nn.ConvTranspose2d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConvTranspose2d(pytorch_module=module,
                                           gradient_variance_reduction_method = gvr,
                                           **mydict)
    elif isinstance(module,nn.ConvTranspose3d):
        gvr = mfvi_arguments["gradient_variance_reduction_method_convolution"]
        mydict = dict(mfvi_arguments)
        del mydict["gradient_variance_reduction_method_linear"],mydict["gradient_variance_reduction_method_convolution"]
        new_module = tpn.VIConvTranspose3d(pytorch_module=module,
                                           gradient_variance_reduction_method = gvr,
                                           **mydict)
    else:
        new_module = module
    return new_module


# Just a convenience function with default wrapping
def convert_to_mfvi_network(network,
                            skip_layer = None,
                            weight_posterior = md.Normal,
                            weight_prior = md.StandardNormalPrior,
                            bias_posterior = md.Normal,
                            bias_prior = md.StandardNormalPrior,
                            train_prior = False,
                            gradient_variance_reduction_method_linear = None,
                            gradient_variance_reduction_method_convolution = None):
    mfvi_arguments = {"weight_posterior" : weight_posterior,
                      "weight_prior" : weight_prior,
                      "bias_posterior" : bias_posterior,
                      "bias_prior" : bias_prior,
                      "train_prior" : train_prior,
                      "gradient_variance_reduction_method_linear" : gradient_variance_reduction_method_linear,
                      "gradient_variance_reduction_method_convolution" : gradient_variance_reduction_method_convolution}
    replacement_rule = partial(MFVI_replacement_rule,mfvi_arguments=mfvi_arguments)
    return replace_layers(network,replacement_rule,skip_layer)
 

def initialize_network(network,
                       variational_posterior_init_method=None,
                       prior_init_method=None):
    for module in network.modules():
        if hasattr(module,'variational_posteriors') and variational_posterior_init_method is not None:
            module.variational_posterior_init(variational_posterior_init_method)
        if hasattr(module,'priors') and prior_init_method is not None:
            module.prior_init(prior_init_method)


def change_gradient_variance_reduction_method(network,
                                              gradient_variance_reduction_method_linear = None,
                                              gradient_variance_reduction_method_convolution = None,
                                              gradient_variance_reduction_method_dropout = None):
    for module in network.modules():
        if isinstance(module,tpn.VILinear):
            if gradient_variance_reduction_method_linear is not None:
                module.gradient_variance_reduction_method = gradient_variance_reduction_method_linear
        elif isinstance(module,tpn.VariationalConvolutionalLayer) or isinstance(module,tpn.VariationalTransposeConvolutionalLayer):
            if gradient_variance_reduction_method_convolution is not None:
                module.gradient_variance_reduction_method = gradient_variance_reduction_method_convolution
        elif isinstance(module,tpn.VariationalDropoutLayer):
            if gradient_variance_reduction_method_dropout is not None:
                module.gradient_variance_reduction_method = gradient_variance_reduction_method_dropout




    
        
