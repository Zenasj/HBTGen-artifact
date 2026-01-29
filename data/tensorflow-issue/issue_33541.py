# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← input shape is inferred as a generic 4D tensor typical for image segmentation

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Placeholder model architecture simulating a segmentation model
        # (User code was based on a segmentation tutorial)
        # We define a simple ConvNet for demonstration
        
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.up1 = tf.keras.layers.UpSampling2D()
        self.conv4 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.up2 = tf.keras.layers.UpSampling2D()
        self.conv5 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.output_layer = tf.keras.layers.Conv2D(1, 1, activation=None)  # logits for binary segmentation
        
        # Loss and optimizer (placeholders must be set externally or in constructor)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        
        # Metrics typically: binary accuracy or IoU-related metrics
        self.metrics_list = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.up1(x)
        x = self.conv4(x)
        x = self.up2(x)
        x = self.conv5(x)
        logits = self.output_layer(x)
        return logits
    
    @property
    def metrics(self):
        # Expose metrics for external use
        return self.metrics_list
    
    @property
    def trainable_weights(self):
        # Return trainable weights from layers
        weights = []
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.output_layer]:
            weights.extend(layer.trainable_weights)
        return weights


class Trainer(object):
    def __init__(self, model, dataloaders, epochs=None, log_interval=1):
        """
        dataloaders: dict {'train': tf.data.Dataset, 'val': tf.data.Dataset}
        model: compiled or defined MyModel instance with optimizer and loss assigned
        epochs: number of epochs to train
        """
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.log_interval = log_interval
    
    def train(self):
        for epoch in range(self.epochs):
            message = '{}/{}:\t'.format(epoch+1, self.epochs)
            
            # Training phase
            self._epoch_train()
            
            # Collect and log train metrics
            for metric in self.model.metrics:
                metric_value = float(metric.result())
                message += 'train_{}: {:.4f}\t'.format(metric.name, metric_value)
                metric.reset_states()

            # Validation phase
            self._epoch_val()
            
            # Collect and log validation metrics
            for metric in self.model.metrics:
                metric_value = float(metric.result())
                message += 'val_{}: {:.4f}\t'.format(metric.name, metric_value)
                metric.reset_states()

            print(message)

    @tf.function
    def _epoch_train(self):
        """
        Perform one training epoch using tf.function for compilation.
        Dataset iteration in tf.function requires a tf.data.Dataset with known structure.
        """
        dataset = self.dataloaders['train']
        
        # Iterate over dataset batches
        for step, (x_batch, y_batch) in dataset.enumerate():
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss_value = self.model.loss(y_batch, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
            # Update metrics
            for metric in self.model.metrics:
                metric.update_state(y_batch, tf.nn.sigmoid(logits))

    def _epoch_val(self):
        """
        Validation phase without @tf.function for flexibility.
        Iterates over validation dataset.
        """
        dataset = self.dataloaders['val']
        for x_batch, y_batch in dataset:
            logits = self.model(x_batch, training=False)
            for metric in self.model.metrics:
                metric.update_state(y_batch, tf.nn.sigmoid(logits))

def my_model_function():
    """
    Return an instance of MyModel, fully initialized.
    """
    model = MyModel()
    # The optimizer, loss, and metrics are already assigned inside MyModel for demo.
    # You could customize these here if desired.
    return model

def GetInput():
    """
    Return a random tensor input matching the input expected by MyModel.
    Assuming input shape typical for segmentation:
    Batch size: random between 1 and 4
    Height: 128
    Width: 128
    Channels: 3 (RGB image)
    dtype: tf.float32
    """
    batch_size = 2
    height = 128
    width = 128
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

