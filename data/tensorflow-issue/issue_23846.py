import numpy as np
import tensorflow as tf

flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS(sys.argv)

class Job(object):
    def __init__(self):
        self.ps_hosts = FLAGS.ps_hosts.split(',')
        self.worker_hosts = FLAGS.worker_hosts.split(',')
        self.job_name = FLAGS.job_name
        self.task_index = FLAGS.task_index
        self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts, 'worker': self.worker_hosts})
        self.server = tf.train.Server(self.cluster, job_name=self.job_name, task_index=self.task_index)
        self.is_chief = (self.task_index == 0 and self.job_name == 'worker')
        worker_prefix = '/job:worker/task:%s' % self.task_index
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=self.cpu_device, cluster=self.cluster,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(len(self.ps_hosts), tf.contrib.training.byte_size_load_fn))
        self.num_ps = self.cluster.num_tasks('ps')
        self.num_worker = self.cluster.num_tasks('worker')

    def data_iter(self, batch_size=1000, file_pattern='./input/part-*'):
        def _parse_function(examples):
            features = {}
            features['label'] = tf.FixedLenFeature([], tf.float32)
            features['user_id'] = tf.FixedLenFeature([1], tf.int64)
            features['item_id'] = tf.FixedLenFeature([1], tf.int64)
            instance = tf.parse_example(examples, features)
            return instance['label'], instance['user_id'], instance['item_id']

        with tf.name_scope('input'):
            files = tf.data.Dataset.list_files(file_pattern)
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                        lambda file: tf.data.TFRecordDataset(file),
                        cycle_length=1, sloppy=True))
            dataset = dataset.prefetch(buffer_size=batch_size*2)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(_parse_function, num_parallel_calls=1)
            iterator = dataset.make_initializable_iterator()
            return iterator

    def model(self, user_id, item_id):
        user_embedding_variable = tf.get_variable('user_emb_var', [1000000, 32], initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5, dtype=tf.float32))
        item_embedding_variable = tf.get_variable('user_emb_var', [500000, 32], initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5, dtype=tf.float32))
        user_embedding = tf.nn.embedding_lookup(user_embedding_variable, user_id)
        item_embedding = tf.nn.embedding_lookup(item_embedding_variable, item_id)
        user_embedding = tf.reshape(user_embedding, [-1, 32])
        item_embedding = tf.reshape(item_embedding, [-1, 32])
        cross = tf.reduce_sum(user_embedding * item_embedding, 1, keep_dims=True)
        bias = tf.get_variable('bias', initializer=tf.constant(np.zeros((1), dtype=np.float32)), dtype=tf.float32)
        layer = cross + bias
        weight_np = np.zeros((1, 2), dtype=np.float32)
        weight_np[:, 1] = 1
        weight = tf.get_variable('weight', initializer=tf.constant(weight_np), dtype=tf.float32, trainable=False)
        logits = tf.matmul(layer, weight)
        return logits

    def train(self):
        if self.job_name == 'ps':
            with tf.device('/cpu:0'):
                self.server.join()
        elif self.job_name == 'worker':
            with tf.Graph().as_default():
                with tf.device(self.param_server_device):
                    train_iterator = self.data_iter()
                    train_label, train_user_id, train_item_id = train_iterator.get_next()
                    train_logit = self.model(train_user_id, train_item_id)
                    train_label = tf.to_int64(train_label)
                    train_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logit, labels=train_label)
                    train_loss = tf.reduce_mean(train_cross_entropy, name='loss')
                    opt = tf.train.AdamOptimizer(learning_rate=0.001)
                    train_op = opt.minimize(train_loss)
                    saver = tf.train.Saver()

                    sess_config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        device_filters=["/job:ps", "/job:%s/task:%d" % (self.job_name, self.task_index)],
                        operation_timeout_in_ms=60000,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
                    with tf.train.MonitoredTrainingSession(master=self.server.target,
                                                           is_chief=self.is_chief,
                                                           config=sess_config) as sess:
                        epoch_num = 0
                        while epoch_num < 10:
                            epoch_num += 1
                            sess.run(train_iterator.initializer)
                            while True:
                                try:
                                    sess.run(train_op)
                                except tf.errors.OutOfRangeError:
                                    saver.save(sess=sess._sess._sess._sess._sess,
                                            save_path="some_hdfs_path/model.checkpoint."+str(epoch_num),
                                            latest_filename='checkpoint.'+str(epoch_num))
                                    break

def main(_):
    job = Job()
    job.train()

if __name__ == '__main__':
    tf.app.run()

server_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
self.server = tf.train.Server(self.cluster, job_name=self.job_name,
                              task_index=self.task_index, config=server_config)

train_user_id_placeholder = tf.placeholder(...)
train_item_id_placeholder = ...
train_label_placeholder = ...
train_logit = self.model(train_user_id_placeholder, train_item_id_placeholder)
...
train_op = opt.minimize(train_loss)
...
try:
  train_user_id_val, train_item_id_val, train_label_val = sess.run([train_user_id, train_item_id, train_label])
  sess.run(train_op,
      feed_dict={
        train_user_id_placeholder: train_user_id_val,
        train_item_id_placeholder: train_item_id_val,
        train_label_placeholder: train_label_val})
except tf.errors.OutOfRangeError:
   ...