# tf.random.uniform((B, 10), dtype=tf.float32)  ← Input shape inferred from model input layer: shape=(None, 10)

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='MyScheduler')
class MyScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super(MyScheduler, self).__init__(**kwargs)

    def __call__(self, step):
        # In the issue, the scheduler simply returns the step as learning rate
        return tf.cast(step, tf.float32)

    def get_config(self):
        return {}

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer matching the original example (input 10 → output 10)
        self.dense = tf.keras.layers.Dense(10)
        
        # Learning rate scheduler and optimizer as per example
        self.lr_scheduler = MyScheduler()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler)
        # Compile model with optimizer and loss so optimizer state is tracked
        self.compile(optimizer=self.optimizer, loss='mse')

    def call(self, inputs, training=None):
        # Simple forward pass
        return self.dense(inputs)

    def save_custom(self, path):
        # Save model using built-in save to preserve optimizer state and lr scheduler with iterations
        self.save(path)

    @classmethod
    def load_custom(cls, path):
        # Load complete model with optimizer and scheduler restored
        return tf.keras.models.load_model(path, custom_objects={'MyScheduler': MyScheduler, 'MyModel': cls})

def my_model_function():
    # Return an instance of MyModel ready for use and training
    return MyModel()

def GetInput():
    # Return a random tensor input matching input shape (batch=16, features=10)
    # batch size 16 is consistent with dataset used in the issue
    return tf.random.uniform((16, 10), dtype=tf.float32)

