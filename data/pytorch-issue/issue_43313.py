for key, param in self._parameters.items():
    self._parameters[key] = param.to(device)