import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class Trainer(tf.keras.Model):
  """A training only model."""

  def __init__(self, model, params):
    super(Trainer, self).__init__()
    self.model = model
    self.params = params
    self._num_replicas_in_sync = tf.distribute.get_strategy(
    ).num_replicas_in_sync

  def call(self, inputs, mode="train"):
    return self.model(inputs, mode)

  def train_step(self, inputs):
    """The logic for one training step."""
    with tf.GradientTape() as tape:
      logits, _, _ = self(inputs, mode="train", training=True)
      targets = models.remove_sos_from_seq(inputs["target_ids"],
                                           self.params.pad_token_id)
      loss = transformer_metrics.transformer_loss(logits, targets,
                                                  self.params.label_smoothing,
                                                  self.params.vocab_size)
      # Scales the loss, which results in using the average loss across all
      # of the replicas for backprop.
      scaled_loss = self.optimizer.get_scaled_loss(loss) / self._num_replicas_in_sync

    tvars = self.trainable_variables
    grads = self.optimizer.get_unscaled_gradients(tape.gradient(scaled_loss, tvars))
    self.optimizer.apply_gradients(list(zip(grads, tvars)))
    return {
        "training_loss": loss,
        "learning_rate": self.optimizer._decayed_lr(var_dtype=tf.float32)
    }


def train(params, strategy, dataset=None):
  """Runs training."""

  if not dataset:
    dataset = input_pipeline.get_input_dataset(
        FLAGS.train_file_pattern,
        FLAGS.train_batch_size,
        params,
        is_training=True,
        strategy=strategy)

  with strategy.scope():
    model = models.create_model(
        FLAGS.model_type, params, init_checkpoint=FLAGS.init_checkpoint)
    opt = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt,"dynamic")
    trainer = Trainer(model, params)
    model.global_step = opt.iterations

    trainer.compile(
        optimizer=opt,
        experimental_steps_per_execution=FLAGS.steps_per_loop)
    summary_dir = os.path.join(FLAGS.model_dir, "summaries")
    summary_callback = tf.keras.callbacks.TensorBoard(
        summary_dir, update_freq=max(100, FLAGS.steps_per_loop))
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=10,
        step_counter=model.global_step,
        checkpoint_interval=FLAGS.checkpoint_interval)
    if checkpoint_manager.restore_or_initialize():
      logging.info("Training restored from the checkpoints in: %s",
                   FLAGS.model_dir)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

  # Trains the model.
  steps_per_epoch = min(FLAGS.train_steps, FLAGS.checkpoint_interval)
  epochs = FLAGS.train_steps // steps_per_epoch
  history = trainer.fit(
      x=dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      callbacks=[summary_callback, checkpoint_callback],
      verbose=2)
  train_hist = history.history
  # Gets final loss from training.
  stats = dict(training_loss=float(train_hist["training_loss"][-1]))
  return stats