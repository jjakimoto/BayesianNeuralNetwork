import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


class LockedDropout(nn.Module):
    """ Dropout with masks fixed along with time axis

    Parameters
    ----------
    dropout: float
        Probability to drop unit
    batch_first: bool
        If Ture, input shape has to be (n_batch, T, dim)
        else, (T, n_batch, dim)
    """

    def __init__(self, dropout=0., batch_first=True):
        super(LockedDropout, self).__init__()
        self.dropout = dropout
        self.batch_first = batch_first
        self.mask = None

    def forward(self, x, fixed=False):
        """Forward

        Paramers
        --------
        x: torch.tensor
        fixed: bool
            If True, use a mask previously sampled

        Returns
        -------
        masked tensor
        """

        if not self.training or not self.dropout:
            return x
        if self.batch_first:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(
                1 - self.dropout)
        else:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(
                1 - self.dropout)
        if self.mask is None or not fixed:
            self.mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = self.mask.expand_as(x)
        return mask * x


class WeightDrop(torch.nn.Module):
    """Dropconnect applied to general weights inside nn.Module

    Parameters
    ----------
    module: nn.Module
    weights: list(str)
        Names of parameters stored in the module
    dropout: float
        Probability to drop units
    """
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout,
                                                            name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, x, hidden=None, fixed=False, *args, **kwargs):
        if not fixed:
            self._setweights()
        return self.module.forward(x, hidden, *args, **kwargs)
