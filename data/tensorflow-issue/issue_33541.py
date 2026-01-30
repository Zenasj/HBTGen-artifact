import tensorflow as tf
class Trainer(object):
    def __init__(self, model, dataloaders, epochs=None,
                 steps=-1, log_interval=1):
        """
        dataloaders: {'train': train_loader, 'val': val_loader}
        Here we assume that the model has been compiled, i.e. it contains
            an optimizer and a loss function
        """
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.log_interval = log_interval
    
    def train(self):
        for epoch in range(self.epochs):
            message = '{}/{}:\t'.format(epoch, self.epochs)

            # Training phase
            self._epoch_train()
            # Tensorboard: https://www.tensorflow.org/tensorboard/get_started
            # Display metrics at the end of each epoch.
            for metric in self.model.metrics:
                metric_value = float(metric.result())
                message += 'train_{}: {}\t'.format(metric.name, metric_value)
                metric.reset_states()

            # Validation phase
            self._epoch_val()
            for metric in self.model.metrics:
                metric_value = float(metric.result())
                message += 'val_{}: {}\t'.format(metric.name, metric_value)
                metric.reset_states()

            print(message)

    @tf.function
    def _epoch_train(self):
        """
        Perform one training epoch
        """
        # Iterate over the batches of the dataset.
        # Note: use ds.enumerate() instead of enumerate(ds)
        # https://github.com/tensorflow/tensorflow/issues/30802
        for step, (x_batch, y_batch) in self.dataloaders['train'].enumerate():
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = self.model(x_batch)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = self.model.loss(y_batch, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            # Update metrics
            for metric in self.model.metrics:
                metric(y_batch, logits)

    def _epoch_val(self):
        """
        Perform one validation epoch
        """
        # Run a validation loop at the end of each epoch.
        for x_batch, y_batch in self.dataloaders['val']:
            logits = self.model(x_batch)
            # Update metrics
            for metric in self.model.metrics:
                metric(y_batch, logits)

@tf.function
def _train_step(self, batch):
    x_batch, y_batch = batch
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = self.model(x_batch)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss = self.model.loss(y_batch, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss, self.model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update metrics
    for metric in self.model.metrics:
        metric(y_batch, logits)
    
    return loss

@tf.function
def _val_step(self, batch):
    x_batch, y_batch = batch
    logits = self.model(x_batch)
    for metric in self.model.metrics:
        metric(y_batch, logits)