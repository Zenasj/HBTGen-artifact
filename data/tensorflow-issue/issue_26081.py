class MobileNetDeepEstimator:
    def __init__(self, image_size, alpha, num_neu, weights=None):

        """if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:"""
        if (1==1):
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
            self.alpha = alpha
            self.num_neu = num_neu
            self.weights = weights
            self.FC_LAYER_SIZE = 1024

    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3,
                                    include_top=False, weights=self.weights, input_tensor=None, pooling=None)

        x = model_mobilenet(inputs)

        feat_a = GlobalAveragePooling2D()(x)
        feat_a = Dropout(0.5)(feat_a)
        feat_a = Dense(self.FC_LAYER_SIZE, activation="relu")(feat_a)

        pred_g_softmax = Dense(2, activation='softmax', name='gender')(feat_a)
        pred_a_softmax = Dense(self.num_neu, activation='softmax', name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[pred_g_softmax, pred_a_softmax])

        return model

hist = tpu_model.fit_generator(
        image_generator.flow(mode='train'),
        steps_per_epoch=int(len(train_keys) / batch_size),
        epochs=nb_epochs,
        callbacks=callbacks,
        validation_data=image_generator.flow('val'),
        validation_steps=int(len(val_keys) / batch_size)
    )

def _get_next_batch(output_generator, mode):
  """Retrieves the next batch of input data."""
  try:
    generator_output = next(output_generator)
  except (errors.OutOfRangeError, StopIteration):
    # Returning `None` will trigger looping to stop.
    logging.warning('Your dataset iterator ran out of data.')
    return None
  if not isinstance(generator_output, tuple):
    if mode == 'predict':
      # Always wrap in a tuple.
      return (generator_output,)
    else:
      raise ValueError('Output of generator should be '
                       'a tuple `(x, y, sample_weight)` '
                       'or `(x, y)`. Found: ' + str(generator_output))

if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))