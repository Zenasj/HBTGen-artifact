import tensorflow as tf

class GridRNNTest(tf.test.TestCase):
    def setUp(self):
        self.num_features = 1
        self.time_steps = 1
        self.batch_size = 1
        tf.reset_default_graph()
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features])
        self.cell = grid_rnn_cell.Grid1LSTMCell(num_units=8)

    def test_simple_grid_rnn(self):
        self.input_layer = tf.unstack(self.input_layer, self.time_steps, 1)
        rnn.static_rnn(self.cell, self.input_layer, dtype=tf.float32)

class BidirectionalGridRNNTest(tf.test.TestCase):
    def setUp(self):
        self.num_features = 1
        self.time_steps = 1
        self.batch_size = 1
        tf.reset_default_graph()
        self.input_layer = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.num_features])
        self.cell_fw = grid_rnn_cell.Grid1LSTMCell(num_units=8)
        self.cell_bw = grid_rnn_cell.Grid1LSTMCell(num_units=8)

    def test_simple_bidirectional_grid_rnn(self):
        self.input_layer = tf.unstack(self.input_layer, self.time_steps, 1)
        rnn.static_bidirectional_rnn(self.cell_fw, self.cell_bw, self.input_layer, dtype=tf.float32)