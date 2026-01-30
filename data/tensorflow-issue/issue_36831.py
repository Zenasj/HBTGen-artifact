import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

train_dir = config.TRAIN_DIR
train_ds = tf.data.Dataset.list_files(str(train_dir / "*"))
train_ds = (
    train_ds.map(load_frames, num_parallel_calls=12)
    .batch(batch_size)
    .prefetch(buffer_size=batch_size)
)

model = MyModel()
# Keep results for plotting
train_loss_results = []
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
for epoch in epochs:
    epoch_loss_avg = tf.keras.metrics.Mean()
    for inputs in train_ds:
        with tf.GradientTape() as tape:
            input_1, input_2, input_3 = inputs
            predictions, warping_output = model(inputs, training=True)
            rec_loss = mse_loss_fn(input_3, predictions)
            
        grads = tape.gradient(rec_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        epoch_loss_avg(grads)  # Add current batch loss

    train_loss_results.append(epoch_loss_avg.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

class MyModel(tf.keras.Model):
    def __init__(self, name="MyModel", **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=7, strides=1, padding="same"
        )

    def call(self, inputs, training=True, **kwargs):
        input_1, input_2, input_3 = inputs
        out = self.conv(input_1)
        return out