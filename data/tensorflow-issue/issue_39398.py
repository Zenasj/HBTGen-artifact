# tf.random.uniform((32, 784), dtype=tf.float32) ← Input shape inferred from example; batch size 32, flattened 28x28 image vector

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # As per the example, the model processes an input vector of size 784 and applies 3 "blocks"
        # Each block: Dense(32, relu) -> Dropout(0.2) -> Dense(10, softmax)
        # Outputs summed cumulatively via lambda layer in original; here we implement sum via tensors
        
        self.blocks = []
        for i in range(3):
            block = {}
            block['dense1'] = tf.keras.layers.Dense(
                32, activation='relu', name=f'Model/Block{i}/relu')
            block['dropout'] = tf.keras.layers.Dropout(
                0.2, name=f'Model/Block{i}/dropout')
            block['dense2'] = tf.keras.layers.Dense(
                10, activation='softmax', name=f'Model/Block{i}/softmax')
            self.blocks.append(block)

    def call(self, inputs, training=False):
        # Initialize outputs as zero tensor like in original code, but dynamically here based on batch size
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, 10], dtype=tf.float32)

        # Iterate over each block and sum their outputs explicitly to avoid TensorFlowOpLayer artifacts
        for i, block in enumerate(self.blocks):
            x = block['dense1'](inputs)
            x = block['dropout'](x, training=training)
            x = block['dense2'](x)
            # Avoid using bare python '+' operator on tensors directly without naming to prevent messy TensorBoard entries
            # Use functional addition with a named Lambda layer alternative by using tf.add here explicitly
            outputs = tf.add(outputs, x, name=f'Model/add/add{i}')
        return outputs

def my_model_function():
    # Return an instance of MyModel; no special initialization required
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape (32, 784) matching the batch size and input vector size from the example
    return tf.random.uniform((32, 784), dtype=tf.float32)

