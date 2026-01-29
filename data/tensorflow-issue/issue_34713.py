# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32), tf.random.uniform((B, None, 10), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Image input branch
        self.conv2d = layers.Conv2D(3, 3)
        self.global_max_pool_2d = layers.GlobalMaxPooling2D()
        # Time series input branch
        self.conv1d = layers.Conv1D(3, 3)
        self.global_max_pool_1d = layers.GlobalMaxPooling1D(name='x2pool')
        # Concatenate and classify
        self.concat = layers.Concatenate()
        self.classifier = layers.Dense(5, activation='softmax', name='class_output')

    def call(self, inputs, training=False):
        # inputs is expected to be a dict with keys 'img_input' and 'ts_input'
        img_input = inputs['img_input']  # shape (B, 32, 32, 3)
        ts_input = inputs['ts_input']    # shape (B, T, 10), T is variable length (None)

        # Image branch forward pass
        x1 = self.conv2d(img_input)           # (B, 30, 30, 3) - conv changes spatial dims from 32->30
        x1pool = self.global_max_pool_2d(x1) # (B, 3)

        # Time series branch forward pass
        x2 = self.conv1d(ts_input)            # (B, T-2, 3) because kernel=3 in conv1d
        x2pool = self.global_max_pool_1d(x2) # (B, 3)

        # Concatenate pooled features from both branches
        x = self.concat([x1pool, x2pool])     # (B, 6)

        # Class output head (softmax)
        class_output = self.classifier(x)     # (B, 5)

        # Original model outputs were in a list:
        # outputs = [x2pool, class_output]
        # This can cause ambiguity with losses keyed by name due to positional indexing.
        # So we return outputs as a dictionary keyed by layer names to match loss dict usage.
        return {
            'x2pool': x2pool,          # pooled conv1d output, no loss attached
            'class_output': class_output
        }

def my_model_function():
    # Return an instance of MyModel. No weights initialization needed beyond default.
    return MyModel()

def GetInput():
    # Return a dict with two tensor inputs matching the expected shapes:
    # 'img_input' shape (B=4, 32, 32, 3), float32
    # 'ts_input' shape (B=4, T=20, 10), float32 (T=20 chosen as example variable length)
    B = 4
    img_input = tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)
    ts_input = tf.random.uniform((B, 20, 10), dtype=tf.float32)
    return {
        'img_input': img_input,
        'ts_input': ts_input,
    }

