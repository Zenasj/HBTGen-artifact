import numpy as np
import tensorflow as tf

def _read_and_batch_from_files(input_arr, target_arr, shuffle=True, shuffle_val=20):

  dataset = tf.data.Dataset.from_tensor_slices((input_arr, target_arr))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=6000, seed=shuffle_val,
                              reshuffle_each_iteration=True)
  else:
    dataset = dataset.batch(base_model_conf_obj.batch_size)
  dataset = dataset.repeat()
  return dataset
 
input_arr_test = np.load('/content/gdrive/My Drive/for_ds/x_test_tok.npy', allow_pickle=True)
target_arr_test = np.load('/content/gdrive/My Drive/for_ds/y_test_tok.npy', allow_pickle=True)
# i used this just for test before use init training/test files
train_dataset =  _read_and_batch_from_files(input_arr_test, target_arr_test, 
                                            shuffle=True, shuffle_val=20)

class SingleDeviceStrategy(object):
  def __enter__(self, *args, **kwargs):
    pass

  def __exit__(self, *args, **kwargs):
    pass

  def scope(self):
    return self

  def experimental_distribute_dataset(self, dataset):
    return dataset

  def experimental_run_v2(self, func, args, kwargs):
    return func(*args, **kwargs)

  def reduce(self, reduction_type, distributed_data, axis):  # pylint: disable=unused-argument
    return distributed_data

def connect_to_tpu(tpu=None):
  if tpu:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
    tf.config.experimental_connect_to_host(cluster_resolver.get_master())
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    return strategy, "/task:1" if os.environ.get("COLAB_TPU_ADDR") else "/job:worker"
  return SingleDeviceStrategy(), ""

worker_tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR']
d_strat, device = connect_to_tpu(worker_tpu)
with d_strat.scope():
  dataset_iter = iter(d_strat.experimental_distribute_dataset(train_dataset))
  for epoch in range(starting_epoch, num_epochs):
    start = time.time()
    inp, tar = next(dataset_iter)
    def tpu_step(inp, tar):
      tar_inp = tar[:, :-1]
      tar_real = tar[:, 1:]
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
      with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                        True,
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask)
        loss = loss_function(tar_real, predictions)
        loss_ce = loss * (1.0 / base_model_conf.batch_size)
      variables = transformer.trainable_variables
      gradients = tape.gradient(loss_ce, variables)
      gradients = [(tf.clip_by_value(grad, -1.0, 1.0))
                        for grad in gradients]
      train_loss.update_state(loss_ce)
      train_op = optimizer.apply_gradients(zip(gradients, variables))
      with tf.control_dependencies([train_op]):
        return tf.cast(optimizer.iterations, tf.float32)

    @tf.function
    def train_step(inp, tar):
      distributed_metric = d_strat.experimental_run_v2(tpu_step, args=[inp, tar])
      step = d_strat.reduce(tf.distribute.ReduceOp.MEAN, distributed_metric, axis=None)
      return step

    step = tf.cast(train_step(inp, tar), tf.int32)
    train_loss.reset_states()
    train_loss_res = train_loss.result().numpy()
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss_res,))
    print('Time taken for train 1 epoch: {} secs\n'.format(time.time() - start))
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    print('Time taken for test 1 epoch: {} secs\n'.format(time.time() - start))