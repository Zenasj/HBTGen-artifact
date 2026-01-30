import tensorflow as tf

def _train_model_default(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, without `DistributionStrategies`.
    Args:
      input_fn: A function that provides input data for training as minibatches.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      Loss from training
    """
    worker_hooks = []
    with ops.Graph().as_default() as g, g.device(self._device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step_tensor = self._create_and_assert_global_step(g)

      # Skip creating a read variable if _create_and_assert_global_step
      # returns None (e.g. tf.contrib.estimator.SavedModelEstimator).
      if global_step_tensor is not None:
        training_util._get_or_create_global_step_read(g)  # pylint: disable=protected-access
      
      features, labels, input_hooks = (
          self._get_features_and_labels_from_input_fn(
              input_fn, model_fn_lib.ModeKeys.TRAIN))
      worker_hooks.extend(input_hooks)
    
      estimator_spec = self._call_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
      global_step_tensor = training_util.get_global_step(g)
      return self._train_with_estimator_spec(estimator_spec, worker_hooks,
                                             hooks, global_step_tensor,
                                             saving_listeners)

estimator_spec = self._call_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)