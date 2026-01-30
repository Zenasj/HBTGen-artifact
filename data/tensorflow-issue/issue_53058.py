3
class BaseDenseAttention(Layer):
    def __init__(self, causal=False, dropout=0.0,
                 **kwargs):
      ...
    
    def call(self,
             inputs,
             mask=None,
             training=None,
             return_attention_scores=False):
      ...