import torch.nn as nn
import torch.nn.functional as F

class WeightDropout(Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module:nn.Module, weight_p:float, layer_names:Collection[str]=['weight_hh_l0']):
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        self.idxs = [] if hasattr(self.module, '_flat_weights_names') else None
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)
            if self.idxs is not None: self.idxs.append(self.module._flat_weights_names.index(layer))
        if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
            self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        "Apply dropout to the raw weights."
        for i,layer in enumerate(self.layer_names):
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)
            if self.idxs is not None: self.module._flat_weights[self.idxs[i]] = self.module._parameters[layer]

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

module = nn.LSTM(5, 2)
dp_module = WeightDropout(module, 0.4)