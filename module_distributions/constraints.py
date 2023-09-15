import torch
from torch.distributions.constraints import Constraint


class PositiveTensor(Constraint):
    """
    This is modification to the positive constraint in
    https://github.com/pytorch/pytorch/blob/master/torch/distributions/constraints.py
    since that positive constraint used a python float which at some point in
    time caused weird errors if we want to check positivity for a tensor.
    This might not be needed in current Pytorch version
    """
    def __init__(self):
        super().__init__()

    def check(self, value):
        return torch.zeros_like(value) < value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += 'lower_bound=0.0'
        return fmt_string
    

positive = PositiveTensor()