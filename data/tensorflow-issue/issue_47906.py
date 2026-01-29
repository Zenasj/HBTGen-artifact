# tf.random.uniform((B, 28, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        return logits

# Custom callback to unfreeze weights at a specific epoch and fix the training function
class UnFreezeWeight(tf.keras.callbacks.Callback):
    def __init__(self, freeze_before_epoch):
        super().__init__()
        self.freeze_before_epoch = freeze_before_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.freeze_before_epoch:
            print('set trainable to True.')
            for layer in self.model.layers:
                layer.trainable = True
            # Rebuild the train function to take into account trainable changes (TF 2.7+ fix)
            self.model.make_train_function(force=True)

def my_model_function():
    model = MyModel()
    # Initially freeze all layers
    for layer in model.layers:
        layer.trainable = False
    return model

def GetInput():
    # Return random float tensor with shape (batch_size, 28, 28)
    # Batch size chosen as 32 to match examples
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

