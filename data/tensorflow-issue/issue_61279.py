# tf.random.uniform((B, 784), dtype=tf.float32) ← Assumed input shape based on model Input(shape=(784,))

import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, f=0.5, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.f = f

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.f * tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model.numpy().item() if tf.is_tensor(self.d_model) else self.d_model,
            "warmup_steps": self.warmup_steps,
            "f": self.f
        }

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple feedforward layer on flattened input 784 to 10 units
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # inputs shape should be (batch_size, 784)
        return self.dense(inputs)

def my_model_function():
    # Instantiate the model and set up optimizer with the custom learning rate schedule
    model = MyModel()
    learning_rate_schedule = CustomSchedule(1024)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-5
    )
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor shaped (batch_size, 784) simulating flattened image/vector input
    batch_size = 32  # typical batch size choice for demonstration
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

