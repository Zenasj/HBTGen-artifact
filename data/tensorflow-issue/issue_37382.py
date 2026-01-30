import random
import tensorflow as tf

signature = (tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="x"),)

@tf.function(input_signature=signature)
def run(x):
  with tf.device("/GPU:0"):
        return tf.nn.relu(tf.matmul(x, tf.random.uniform((3, 1))))

concrete_run = run.get_concrete_function()

@tf.nondifferentiable_batch_function(1, 6, 10)
def batch_run(x):
    return concrete_run(x)

class Model(tf.Module):
    @tf.function(input_signature=signature)
    def run(self, x):
        return batch_run(x)

model = Model()
x = tf.random.uniform((3, 3))

# MatMul and Relu are executed on CPU
model.run(x)

concrete_model_run = model.run.get_concrete_function()

# MatMul and Relu are executed on CPU
concrete_model_run(x)

tf.saved_model.save(model, "model", signatures={ "run": model.run })
model = tf.saved_model.load("model")

# MatMul and Relu are executed on GPU
model.signatures["run"](x=x)