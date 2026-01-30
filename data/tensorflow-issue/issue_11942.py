# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris

import tensorflow as tf
print(tf.__version__)
import os

def combine_inputs(X):
    res =  tf.matmul(X, W) + b
    res = tf.identity(res, name="linear_out")


def inference(X):
    return tf.nn.softmax(combine_inputs(X), "softmax_out")


def loss(X, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=combine_inputs(X), name="softmax_entropy"), name="loss")


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label =\
        read_csv(100, "iris.csv", [[0.0], [0.0], [0.0], [0.0], [""]])
    # convert class names to a 0 based class index.
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))
    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))
    return features, label_number


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# Launch the graph in a session, setup boilerplate
with tf.Graph().as_default():
    with tf.Session() as sess:
        # this time weights form a matrix, not a column vector, one "weight vector" per class.
        W = tf.Variable(tf.zeros([4, 3]), name="weights")
        # so do the biases, one per class.
        b = tf.Variable(tf.zeros([3], name="bias"))
        tf.global_variables_initializer().run()
        X, Y = inputs()
        total_loss = loss(X, Y)
        train_op = train(total_loss)
        summary_out = os.path.join(os.path.dirname(__file__), "tf_summary")
        import shutil
        shutil.rmtree(summary_out)
        writer = tf.summary.FileWriter(summary_out, graph=sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # actual training loop
        training_steps = 1000
        for step in range(training_steps):
            sess.run([train_op])
            # for debugging and learning purposes, see how the loss gets decremented thru training steps
            if step % 10 == 0:
                print("loss: ", sess.run([total_loss]))
        evaluate(sess, X, Y)
        coord.request_stop()
        coord.join(threads)
        writer.close()