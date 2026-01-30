import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=0.1, end_lr=0.001, warmup_steps=390*5):
        super(CustomSchedule, self).__init__()

        self.base_lr = base_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
    
    def __call__(self, step=390*35):
        warmup_lr_schedule = tf.linspace(0., self.base_lr, self.warmup_steps)
        iters = tf.range(step, dtype=tf.float32) 
        cosine_lr_schedule = tf.convert_to_tensor([self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1 + \
                        tf.math.cos(tf.constant(math.pi) * t / (step))) for t in iters])
        lr_schedule = tf.concat([warmup_lr_schedule, cosine_lr_schedule], axis=0)
        
        return lr_schedule