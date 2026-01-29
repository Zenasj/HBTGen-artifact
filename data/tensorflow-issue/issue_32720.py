# tf.ones((256, 10), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Variables analogous to get_variable with resource vars:
        # weight shape [10, 10], bias shape [1, 10]
        # Using tf.Variable with initializer similar to Xavier initializer
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.weight = tf.Variable(initializer(shape=(10, 10)), name='DummyWeight', trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=(1, 10)), name='DummyBiasRel', trainable=True)
        
        # BatchNorm layer from tf.keras.layers recommended over tf.layers.batch_normalization
        # fused=True is default true for 2D inputs
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='DummyBatchNorm')
    
    def call(self, inputs, training=False):
        # inputs shape assumed [batch_size, 10]
        batch_size = tf.shape(inputs)[0]
        
        # Reshape inputs to ensure [batch_size, -1] (already that shape)
        output = tf.reshape(inputs, [batch_size, -1])  # shape: [batch_size, 10]

        # Broadcast weights to [batch_size, 10, 10]
        weight_bc = tf.broadcast_to(self.weight, [batch_size, 10, 10])
        
        # Define a matmul function for map_fn: each element is pair (output[i], weight_bc[i])
        def matmul(pair):
            # pair[0] shape [10], pair[1] shape [10, 10]
            # Add batch dim 1:
            inp_exp = tf.expand_dims(pair[0], 0)       # [1, 10]
            prod = tf.matmul(inp_exp, pair[1])          # [1, 10]
            return tf.squeeze(prod, axis=0)              # [10]
        
        # Map the matmul over batch dimension
        output_matmul = tf.map_fn(matmul, (output, weight_bc), dtype=tf.float32)

        # Add bias: bias shape [1, 10], broadcast to [batch_size, 10]
        output_biased = output_matmul + self.bias
        
        # Apply batch normalization
        output_bn = self.batch_norm(output_biased, training=training)
        
        return output_bn

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input expected:
    # Shape [256, 10], dtype float32, as in original code
    return tf.ones(shape=(256, 10), dtype=tf.float32)

