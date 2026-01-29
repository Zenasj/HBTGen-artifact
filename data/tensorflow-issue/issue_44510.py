# tf.random.uniform((B, 784), dtype=tf.float32) ← Assuming batch size B is dynamic and input shape is (784,)
import tensorflow as tf

class DenseOnGPU(tf.keras.layers.Layer):
    def __init__(self, gpu_index, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.gpu_index = gpu_index
        # Create dense layer on assigned GPU device scope
        with tf.device(f"/GPU:{self.gpu_index}"):
            self.dense = tf.keras.layers.Dense(units, activation=activation)

    def call(self, inputs):
        # Forward pass on assigned GPU
        with tf.device(f"/GPU:{self.gpu_index}"):
            return self.dense(inputs)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model parallelism by assigning different layers to different GPUs
        # This mimics the suggested approach from the issue comments
        self.dense_gpu0_1 = DenseOnGPU(0, 256, activation='relu')
        self.dense_gpu1_1 = DenseOnGPU(1, 256, activation='relu')
        self.dense_gpu2_1 = DenseOnGPU(2, 256, activation='relu')

        # The final layers run on GPU:3 and take input passed through previous GPUs
        self.dense_gpu3_1 = DenseOnGPU(3, 256, activation='relu')
        self.dense_gpu3_2 = DenseOnGPU(3, 256, activation='relu')
        self.output_layer = DenseOnGPU(3, 10)  # logits output

    def call(self, inputs):
        # Step through layers, moving tensors as needed across devices implicitly by TF
        x0 = self.dense_gpu0_1(inputs)
        x1 = self.dense_gpu1_1(x0)
        x2 = self.dense_gpu2_1(x1)
        x3 = self.dense_gpu3_1(x2)
        x3 = self.dense_gpu3_2(x3)
        outputs = self.output_layer(x3)
        return outputs

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Provide a random input tensor matching expected input shape (batch size arbitrary, features=784)
    B = 32  # Assume batch size 32 for input generation
    return tf.random.uniform((B, 784), dtype=tf.float32)

