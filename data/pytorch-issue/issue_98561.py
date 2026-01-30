def _momentum_update_key_encoder(self):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)