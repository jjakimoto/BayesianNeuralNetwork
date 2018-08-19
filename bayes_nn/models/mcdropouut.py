import torch
import torch.nn as nn

from ..layers import LockedDropout, WeightDrop


class MCDropoutLSTM(nn.Module):
    """MCDropout for LSTM


    Parameters
    ----------
    n_input: int
    n_hidden: int
    n_output: int
    n_layers: int
        The number of lstm layers
    idrop: float
        Probability to drop input
    drop: float
        Probability to drop output for lstm
    wdrop float
        Probability to drop recurrenct weight
    batch_first: bool
    """
    def __init__(self, n_input, n_hidden, n_output=1, n_layers=2,
                 idrop=.25, drop=.25, wdrop=.25, batch_first=True):
        super(MCDropoutLSTM, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.idrop = idrop
        self.drop = drop
        self.wdrop = wdrop
        self.rnns = [
            nn.LSTM(n_input if i == 0 else n_hidden,
                    n_hidden, num_layers=1, batch_first=batch_first)
            for i in range(n_layers)
        ]
        # Fixed Dropout
        self.inputdrop = LockedDropout(idrop, batch_first)
        self.lockdrops = [LockedDropout(drop, batch_first) for i in
                          range(n_layers)]
        # Recurrent weight dropout
        if self.wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                         for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.output_layer = nn.Linear(n_hidden, n_output)

    def _forward(self, x, hidden=None, fixed=False):
        x = self.inputdrop(x, fixed=fixed)
        new_hs = []
        new_cs = []
        for i, rnn in enumerate(self.rnns):
            if hidden is None:
                hidden_i = None
            else:
                hidden_i = (hidden[0][[i]], hidden[1][[i]])
            x, (new_h, new_c) = rnn(x, hidden=hidden_i, fixed=fixed)
            x = self.lockdrops[i](x, fixed=fixed)
            new_hs.append(new_h)
            new_cs.append(new_c)
        new_hs = torch.cat(new_hs, 0)
        new_cs = torch.cat(new_cs, 0)
        output = self.output_layer(x)
        return output, (new_hs, new_cs)

    def forward(self, x, hidden=None, forward=0):
        output, hidden = self._forward(x, hidden, fixed=False)
        if forward > 0:
            preds = []
            pred_input = output[:, [-1]]
            pred_hidden = hidden
            for i in range(forward):
                pred, pred_hidden = self._forward(pred_input, pred_hidden,
                                                  fixed=True)
                preds.append(pred)
                pred_input = pred
            preds = torch.cat(preds, 1)
            hidden = pred_hidden
            output = torch.cat([output, preds], 1)
        return output, hidden

    def predict(self, x, hidden=None, forward=0, T=10):
        """Predict mean and standard deviation through MC Dropout

        Parameters
        ----------
        x: torch.tensor
        hidden: tuple of hidden states, optional
        forward: int, (default 0)
            How many forward to predict
        T: int
            The number of trials to estimate mean and standard
            deviation

        Returns
        -------
        pred_mean: torch.tensor
            Empirical mean
        pred_std: torch.tensor
            Empirical standard deviation
        """
        outputs = []
        self.train()
        for t in range(T):
            output = self(x, hidden, forward=forward).detach().numpy()
            outputs.append(output)
        pred_mean = np.mean(outputs, axis=0)
        pred_std = np.std(outputs, axis=0)
        return pred_mean, pred_std