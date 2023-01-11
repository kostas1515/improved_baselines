import torch.nn as nn
import torch
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