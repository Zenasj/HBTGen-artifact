# tf.random.uniform((B,)) ← The learning rate schedule expects a scalar or 1D tensor step input

import math
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example dummy layer to form a simple model
        # (Model itself not focus here; main logic is the LR schedule)
        self.dense = tf.keras.layers.Dense(1)
        
        # Custom learning rate schedule embedded as a submodule
        self.custom_schedule = self.CustomSchedule()

    def call(self, inputs):
        # Forward pass dummy logic: apply dense layer
        return self.dense(inputs)
    
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr=0.1, end_lr=0.001, warmup_steps=390*5, total_steps=390*35):
            super().__init__()
            self.base_lr = base_lr
            self.end_lr = end_lr
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.total_steps = tf.cast(total_steps, tf.float32)

        def __call__(self, step):
            # Ensure step is float32 tensor
            step = tf.cast(step, tf.float32)

            # Warmup: linear increase from 0 to base_lr in warmup_steps steps
            # For steps beyond warmup_steps, use cosine decay
            # Learning rate at step = 
            #   if step < warmup_steps:
            #       step / warmup_steps * base_lr
            #   else:
            #       end_lr + 0.5 * (base_lr - end_lr) * (1 + cos(pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            
            def warmup_lr():
                return (step / self.warmup_steps) * self.base_lr

            def decay_lr():
                decay_steps = self.total_steps - self.warmup_steps
                decay_fraction = (step - self.warmup_steps) / decay_steps
                cosine_decay = 0.5 * (1 + tf.cos(math.pi * decay_fraction))
                lr = self.end_lr + (self.base_lr - self.end_lr) * cosine_decay
                return lr
            
            learning_rate = tf.cond(
                step < self.warmup_steps,
                true_fn=warmup_lr,
                false_fn=decay_lr,
            )
            return learning_rate

def my_model_function():
    # Return an instance of MyModel, initializing weights normally
    return MyModel()

def GetInput():
    # The model expects some input to pass through the dense layer.
    # Original issue focuses on learning rate schedule, not input shape.
    # Let's assume input batch size = 4, feature dim = 10 as a reasonable test.
    batch_size = 4
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

