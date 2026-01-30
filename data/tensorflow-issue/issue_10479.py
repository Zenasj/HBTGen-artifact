import tensorflow as tf

class ToyModel(object):
	def __init__(self):
		x = tf.get_variable("x", shape=[5, 3, 7],
							initializer=tf.random_normal_initializer(),
							trainable=False)
		cell = tf.contrib.rnn.LSTMCell(7, use_peepholes=True, initializer=tf.orthogonal_initializer)
		self.rnn_out, self.final_state = tf.nn.dynamic_rnn(cell=cell,
														   inputs=x,
														   parallel_iterations=8,
														   time_major=True,
														   dtype=tf.float32)


graph_context = tf.Graph()
with graph_context.as_default():
	m1 = ToyModel()

	tf_init = tf.global_variables_initializer()
	save_dir = "/Users/delkind/Desktop/whd/tf_checkpoints/unit_test"

	sv = tf.train.Supervisor(logdir=save_dir)
	with sv.managed_session() as sess:
		y1 = m1.rnn_out.eval(session=sess)

		print(y1)

initializer=tf.orthogonal_initializer

use_peephole=True