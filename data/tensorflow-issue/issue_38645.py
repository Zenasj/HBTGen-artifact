import tensorflow as tf

def gen(): 
  for i in itertools.count(1): 
    yield (i, [1] * i) 
ds = tf.data.Dataset.from_tensor_slices(list(range(24)))
ds = ds.interleave(lambda gen_idx: tf.data.Dataset.from_generator(gen,
                                                                  output_types=(tf.float32),
                                                                  args=(gen_idx,),
                                                                  output_shapes=(tf.TensorShape([]), tf.TensorShape([None]))),
                   cycle_length=24,
                   block_length=1,
                   num_parallel_calls=24)

class ValidationCallback(keras.callbacks.Callback):

    def __init__(self, validation_gen, prefix, file_writer):
        super(ValidationCallback, self).__init__()
        config = get_config()
        self.config = config
        self.validation_gen = validation_gen
        self.prefix = prefix
        self.file_writer = file_writer

    def on_batch_end(self, batch, logs=None):
        if batch % 1000 == 0 and batch != 0:
            logger.info('')
            logger.info(f'learning_rate {keras.backend.eval(self.model.optimizer._decayed_lr(tf.float32))}')
            logger.info('')

    def on_epoch_end(self, epoch, logs=None):
        # with a Sequential model
        metrics = self.model.evaluate(self.validation_gen,
                                      verbose=1,
                                      use_multiprocessing=False if self.config['train'].as_bool('use_dataset') else True,
                                      workers=int(self.config['train']['num_workers']) if not self.config['train'].as_bool('use_dataset') else 1)
        with self.file_writer.as_default():
            for name, metric in zip(self.model.metrics_names, metrics):
                logger.info('metric name {} prefix {}'.format(name, self.prefix))
                tf.summary.scalar('{}_{}'.format(self.prefix, name), metric, step=epoch)
                logs['{}_{}'.format(self.prefix, name)] = metric
        logger.info('flushing file writer')

file_writer = tf.summary.create_file_writer(tensorboard_directory + "/metrics")