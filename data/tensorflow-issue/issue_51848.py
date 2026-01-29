# tf.random.uniform((B, 512), dtype=tf.float32)
import tensorflow as tf
import tensorflow_model_optimization as tfmot

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply

class MyModel(tf.keras.Model):
    """
    A model mimicking the 'toy_model' from the issue:
    - Inputs: 
        input_embeddings: shape (None, 512), float32 tensor
        is_training: bool scalar tensor to control training behavior (dropout)
    - Architecture:
        Dropout(rate=0.5) applied to input_embeddings (dropout active only if is_training=True)
        Dense(512) applied on dropout output
    - Quantization-aware mode embeds quantization annotation on Dense layer.
    
    This combined model contains logic to handle training mode and dropout properly,
    reflecting the issue context and intended usage.
    """
    def __init__(self, quantization_aware=False):
        super().__init__()
        self.quantization_aware = quantization_aware
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        if quantization_aware:
            # Annotate the Dense layer for quantization-aware training
            self.dense = quantize_annotate_layer(
                tf.keras.layers.Dense(units=512, name="dense")
            )
            # Apply quantization wrappers after model construction
            # We'll build a sub-model inside construct call and wrap after init call below
            # But since we need a single functional Model, handle quantization_apply externally
            # For this class-based model, we emulate without quantize_apply wrapper 
            # as it requires a functional model wrapper.
            # Instead, here it is a placeholder - caller can wrap externally if needed.
            # So we keep dense as annotated layer.
        else:
            self.dense = tf.keras.layers.Dense(units=512, name="dense")
        
    def call(self, inputs, training=False):
        """
        inputs: tuple/list of two tensors
          - input_embeddings: [batch_size, 512], tf.float32
          - is_training: scalar bool tensor or Python bool indicating if model runs in training mode
        training param is ignored because dropout training/control is driven via is_training input.
        
        Returns output tuple:
          - dropout output tensor
          - dense output tensor after dropout
        """
        input_embeddings, is_training = inputs
        
        # Apply dropout with the is_training flag controlling dropout behavior.
        # This matches the issue's lesson that dropout training argument should be controlled
        # at model call time, not passed explicitly during layer construction.
        x = self.dropout(input_embeddings, training=is_training)
        out = self.dense(x)
        return x, out

def my_model_function(quantization_aware=False):
    """
    Returns an instance of MyModel.
    For quantization_aware=True, the dense layer is annotated for quantization,
    but note that proper quantization wrapping with quantize_apply requires 
    functional model construction outside this class.
    """
    return MyModel(quantization_aware=quantization_aware)

def GetInput():
    """
    Returns a tuple of:
    - input_embeddings: random float32 tensor shaped (16, 512)
    - is_training: scalar bool tensor, set True here for training mode to activate dropout
    
    This matches the inputs expected by MyModel.
    """
    input_embeddings = tf.random.uniform(shape=(16, 512), dtype=tf.float32)
    is_training = tf.constant(True)  # simulate training mode to activate dropout
    return (input_embeddings, is_training)

