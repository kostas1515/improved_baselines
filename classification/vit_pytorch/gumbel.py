import torch.nn as nn
import torch
import numpy as np
class Gumbel(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Gumbel}(x) = \gamma(x) = \frac{1}{\exp\exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Gumbel()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return torch.exp(-torch.exp(-torch.clamp(input,min=-4,max=10)))
    

class NormGumbel(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Gumbel}(x) = \gamma(x) = \frac{1}{\exp\exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = NormGumbel()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        a = self.scale*torch.exp(-torch.exp(-torch.clamp(input,min=-4,max=10)))
        return a/torch.sum(a,dim=-1,keepdims=True)
        
class SoftGumbel(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Gumbel}(x) = \gamma(x) = \Softmax(\frac{s}{\exp\exp(-x)})


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SoftGumbel()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        a = self.scale*torch.exp(-torch.exp(-torch.clamp(input,min=-4,max=10)))
        return a.softmax(dim=-1)
    

class SoftEntropyGumbel(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Gumbel}(x) = \gamma(x) = \Softmax(\frac{s}{\exp\exp(-x)})


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = SoftGumbel()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, scale=16.0):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        base=input.shape[1]
        a = self.scale*torch.exp(-torch.exp(-torch.clamp(input,min=-4,max=10)))
        dist =  a.softmax(dim=-1)
        gain = 1/(-torch.sum(dist * torch.log(dist),keepdims=True)/np.log(base))
        gain[torch.isinf(gain)]==1.0
        gain[torch.isnan(gain)]==1.0
        
        return gain*dist
        