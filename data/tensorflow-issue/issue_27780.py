import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import functools
from absl import app
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

def conv_kernel_initializer(shape, dtype=None, partition_info=None):

    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)
def dense_kernel_initializer(shape, dtype=None, partition_info=None):

    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)

class FakeImageDataInput(object):

    def __init__(self,image_size=64,dataset_size=128*64,use_float16=True):
        self._image_size = image_size
        self._use_float16 = use_float16
        self._dataset_size = dataset_size

    def set_shapes(self, batch_size, images, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))
        return images, labels

    def _get_random_input(self, data):
        return tf.zeros([self._image_size, self._image_size, 3], tf.float16
                                if self._use_float16 else tf.float32)

    def make_source_dataset(self):
        """See base class."""
        return tf.data.Dataset.range(self._dataset_size).repeat().map(self._get_random_input)
    def dataset_parser(self, value):
        #random = tf.random.uniform(shape=[1],dtype=tf.int32,minval=0,maxval=1000 - 1)
        random = np.random.randint(0,999)
        return value, tf.constant(int(random), tf.int32)

    def input_fn(self, params):

        batch_size = params['batch_size']

        dataset = self.make_source_dataset()
        
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=4, drop_remainder=True))

        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset


class TestBlock(object):
    
    def __init__(self,
                 kernel_size,
                 input_filters,
                 output_filters,
                 stride=1,
                 use_depthwise_conv=True):
        self._use_depthwise_conv = use_depthwise_conv
        self._kernel_size = kernel_size
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._stride = stride
        self._build()
    
    def _build(self):
        self._expand_conv = tf.layers.Conv2D(filters=self._input_filters * 6,
                                             kernel_size=[1,1],
                                             strides=[self._stride,self._stride],
                                             kernel_initializer=conv_kernel_initializer,
                                             padding='same',
                                             use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(axis=-1,
                                                 momentum=0.99,
                                                 epsilon=1e-3,
                                                 fused=True)
        if self._use_depthwise_conv:
            self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self._kernel_size,
                strides=[self._stride,self._stride],
                depthwise_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
        else:
            self._depthwise_conv = tf.layers.Conv2D(
                filters=self._input_filters * 6,
                kernel_size=self._kernel_size,
                strides=[self._stride,self._stride],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(axis=-1,
                                                 momentum=0.99,
                                                 epsilon=1e-3,
                                                 fused=True)
        self._project_conv = tf.layers.Conv2D(filters=self._output_filters,
                                             kernel_size=[1,1],
                                             strides=[self._stride,self._stride],
                                             kernel_initializer=conv_kernel_initializer,
                                             padding='same',
                                             use_bias=False)
        self._bn2 = tf.layers.BatchNormalization(axis=-1,
                                                 momentum=0.99,
                                                 epsilon=1e-3,
                                                 fused=True)
    def call(self,inputs,training=True):
        x = tf.nn.relu(self._bn0(self._expand_conv(inputs), training=training))
        x = tf.nn.relu(self._bn1(self._depthwise_conv(x), training=training))
        x = self._bn2(self._project_conv(x), training=training)
        return x

class TestModel(tf.keras.Model):
    
    def __init__(self,use_depthwise_conv=True):
        super(TestModel,self).__init__()
        self._use_depthwise_conv = use_depthwise_conv
        self._build()
    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float32,
                           *args, **kwargs):
        if dtype is tf.float16:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self,scope):
        return tf.variable_scope(scope,custom_getter=self._custom_dtype_getter)
    
    def _build(self):
        self._conv_stem = tf.layers.Conv2D(filters=32,
                                           kernel_size=[3,3],
                                           strides=[2,2],
                                           kernel_initializer=conv_kernel_initializer,
                                           padding='same',
                                           use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(axis=-1,
                                                 momentum=0.99,
                                                 epsilon=1e-3,
                                                 fused=True)
        self._conv_head = tf.layers.Conv2D(filters=1280,
                                           kernel_size=[1,1],
                                           strides=[1,1],
                                           kernel_initializer=conv_kernel_initializer,
                                           padding='same',
                                           use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(axis=-1,
                                                 momentum=0.99,
                                                 epsilon=1e-3,
                                                 fused=True)
        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_last')
        self._fc = tf.layers.Dense(
            1000,
            kernel_initializer=dense_kernel_initializer)
        self._blocks = []
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=32,output_filters=16,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=16,output_filters=24,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=24,output_filters=24,stride=2,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=24,output_filters=40,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=40,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=40,stride=2,use_depthwise_conv=self._use_depthwise_conv))
        #self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=80,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        #self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=80,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=80,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=40,output_filters=80,stride=2,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=80,output_filters=112,stride=1,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=112,output_filters=160,stride=2,use_depthwise_conv=self._use_depthwise_conv))
        self._blocks.append(TestBlock(kernel_size=[3,3],input_filters=160,output_filters=320,stride=1,use_depthwise_conv=self._use_depthwise_conv))


    def call(self,inputs,training=True):
        with self._model_variable_scope('testmode'):
            outputs = None
            self.endpoints = {}
            with tf.variable_scope('testmode_stem'):
                outputs = tf.nn.relu(
                    self._bn0(self._conv_stem(inputs), training=training))
            self.endpoints['stem'] = outputs

            for idx, block in enumerate(self._blocks):
                with tf.variable_scope('mnas_blocks_%s' % idx):
                    outputs = block.call(outputs, training=training)
                    self.endpoints['block_%s' % idx] = outputs
            self.endpoints['global_pool'] = outputs
            with tf.variable_scope('testmode_head'):
                outputs = tf.nn.relu(self._bn1(self._conv_head(outputs), training=training))
                outputs = self._avg_pooling(outputs)
                outputs = self._fc(outputs)
                self.endpoints['head'] = outputs
        return outputs,self.endpoints


def test_model_fn(features, labels, mode, params):
    
    model = TestModel(use_depthwise_conv=params['use_depthwise_conv'])
    if params['use_float16']:
        logits, _  = model(features,training=True)
        logits = tf.cast(logits, tf.float32)
    else:
        logits, _  = model(features,training=True)
    

    weight_decay = 1e-5
    one_hot_labels = tf.one_hot(labels, 1000)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        label_smoothing=0.1)
    loss = cross_entropy + weight_decay * tf.add_n([
      tf.nn.l2_loss(tf.cast(v,tf.float32))
      for v in tf.trainable_variables()
      if 'batch_normalization' not in v.name
    ])

    learning_rate = 0.01
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.train.get_global_step()
    
    if params['use_float16']:
      
      #loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(4096)
      #loss_scale_optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
      #train_op = loss_scale_optimizer.minimize(loss,tf.train.get_global_step())
      loss_scale = 128
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
      train_op = tf.group(minimize_op, update_ops)

    else: 
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, tf.train.get_global_step())


    tf.summary.scalar('cross_entropy',cross_entropy)
    tf.summary.scalar('loss',loss)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=None,
      scaffold=None)

def main(unused_argv):
    use_float16 = True
    use_depthwise_conv = True
    model_dir = 'D:/tf_project/depthwiseconv_mixpt/model/'
    batch_size = 64
    train_steps = 100000
    distribution_strategy = None

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.estimator.RunConfig(
      model_dir=model_dir,
      train_distribute=distribution_strategy,
      save_checkpoints_steps=1000,
      log_step_count_steps=1,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True)),
          #gpu_options=gpu_options
          ),
    )
    params = dict(
        use_depthwise_conv=use_depthwise_conv,
        steps_per_epoch=64,
        use_float16=use_float16,
        batch_size=batch_size
        )
    testmode_est = tf.estimator.Estimator(
      model_fn=test_model_fn,
      config=config,
      params=params
    )

    fake_imagenet_train = FakeImageDataInput(image_size=224,dataset_size= 128 * batch_size,use_float16=use_float16)
    current_step = load_global_step_from_checkpoint_dir(  # pylint: disable=protected-access
        model_dir)

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', train_steps,
        train_steps / 64, current_step)
    testmode_est.train(
        input_fn=fake_imagenet_train.input_fn,
        max_steps=train_steps
    )

def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)