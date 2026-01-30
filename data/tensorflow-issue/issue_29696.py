import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

uniform_regularizer=0
model=tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(
    units=5,
    kernel_initializer='he_normal',
    recurrent_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l1(l=uniform_regularizer),
    use_bias=False,
    stateful=True,
    batch_input_shape=(3, 20, 3)),
    tf.keras.layers.Dense(3,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l1(l=uniform_regularizer),
        use_bias=False)
])
def loss(explanvar, targetvar):
  return tf.keras.metrics.mean_absolute_error(
    explanvar,
    targetvar)
model.compile(
    optimizer='adam',
    loss = loss)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True)
history = model.fit(train, 
                    epochs=55, 
                    steps_per_epoch=10,
                    callbacks=[cp_callback],
                    class_weight={0:0.4,
                                  1:0.2,
                                  2:0.4},                   
                    validation_data=test)