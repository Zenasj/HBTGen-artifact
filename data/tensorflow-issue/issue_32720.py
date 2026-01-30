3
import tensorflow as tf


class DummyModelRunner(object):
    def __init__(self):
        self.optimizer = tf.train.AdamOptimizer()

        with tf.variable_scope('Dummy_Vars', use_resource=True):
            self.variables = self.create_variables()

        self.is_train = tf.placeholder_with_default(False, shape=[], name='is_train')

        self.batch_inputs = tf.ones(shape=[256, 10], dtype=tf.float32, name='e1_embs')
        actual_answers = tf.ones(shape=[256, 10, 5], dtype=tf.float32, name='actual_answers')
        targets = tf.ones(shape=[256, 5], dtype=tf.float32, name='targets')

        self.batch_outputs = self.create_predictions(input_embs=self.batch_inputs)

        self.predictions = self.compute_likelihoods(self.batch_outputs, actual_answers)

        self.loss = self.create_loss(self.predictions, targets)

        # The following control dependency is needed in order for batch
        # normalization to work correctly.
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.optimizer.minimize(self.loss)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variables))

    def create_variables(self):
        """
        create all network variables
        :return: all variable dictionary
        """
        weight = tf.get_variable(name='DummyWeight',
                                 shape=[10, 10],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name='DummyBiasRel',
                               shape=[1, 10],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

        variables = {'weights': weight,
                     'biases': bias}

        return variables


    def create_predictions(self, input_embs):
        weight, bias = self.variables['weights'], self.variables['biases']
        is_train_batch_norm = self.is_train

        def matmul(pair):
            input_tensor = pair[0][None]
            projection_tensor = pair[1]
            projection = tf.matmul(input_tensor, projection_tensor)[0]
            return (projection, tf.zeros([]))

        output = tf.reshape(input_embs, [tf.shape(input_embs)[0], -1])
        # Next two lines cause problem
        weight = tf.broadcast_to(weight, [256, 10, 10])
        output = tf.map_fn(fn=matmul, elems=(output, weight))[0] + bias

        output = tf.layers.batch_normalization(output, momentum=.1, reuse=tf.AUTO_REUSE,
                                               training=is_train_batch_norm, fused=True,
                                               name='DummyBatchNorm')

        return output

    def compute_likelihoods(self, input_vectors, actual_answers):
        with tf.name_scope('output_layer'):

            predictions = tf.matmul(input_vectors[:, None, :], actual_answers)[:, 0, :]

        return predictions

    def create_loss(self, predictions, targets):
        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.compat.v1.losses.sigmoid_cross_entropy(targets, predictions),
                name='loss')
        return loss


if __name__ == '__main__':
    # Create the model.
    # Can be /GPU:0 for ubuntu
    with tf.device('/CPU:0'):
        # We are using resource variables because due to some implementation details, this allows us to
        # better utilize GPUs while training.
        with tf.variable_scope('variables', use_resource=True):

            model = DummyModelRunner()

    # Create a TensorFlow session and start training.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    saver = tf.train.Saver()

    # Initialize the values of all variables and the train dataset iterator.
    session.run(tf.global_variables_initializer())

    for step in range(100):
        # print('Hi!')
        feed_dict = {model.is_train: True}

        loss, _ = session.run((model.loss, model.train_op), feed_dict)

        print(loss)
        exit()