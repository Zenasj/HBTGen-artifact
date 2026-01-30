import tensorflow as tf

python
model_variables = my_model.get_variables_list()
optimizer_slots = [
    optimizer.get_slot(var, name)
    for name in optimizer.get_slot_names()
    for var in model_variables
]
all_variables = [
    *model_variables,
    *optimizer_slots,
    global_step,
]
init_op = tf.variables_initializer(all_variables)

python
model_variables = my_model.get_variables_list()
optimizer_slots = [
    optimizer.get_slot(var, name)
    for name in optimizer.get_slot_names()
    for var in model_variables
]
if isinstance(optimizer, tf.train.AdamOptimizer):
    optimizer_slots.extend([
        optimizer._beta1_power, optimizer._beta2_power
    ])
all_variables = [
    *model_variables,
    *optimizer_slots,
    global_step,
]
init_op = tf.variables_initializer(all_variables)

py
def _get_beta_accumulators(self):
    return self._beta1_power, self._beta2_power

def main(_):
    x_minibatch, y_minibatch, y_lengths_minibatch = construct_data_pipeline()
    model = import_model()
    train(model=model, x_minibatch=x_minibatch, y_minibatch=y_minibatch, y_lengths_minibatch=y_lengths_minibatch)

class Model(object):
    def __init__(self, batch_size, initial_learning_rate):
        self.x, self.y_actual, self.y_actual_lengths = self._define_inputs(batch_size)
        self.encoder_outputs, self.encoder_final_states, self.decoder_outputs, self.y_predicted_logits = self._define_model()
        self.loss = self._define_loss()
        self.metrics = self._define_metrics()
        self.optimizer = self._define_optimizer(initial_learning_rate)

def train(model, x_minibatch, y_minibatch, y_lengths_minibatch):
    with tf.Session() as sess:

        # create coordinator to handle threading
        coord = tf.train.Coordinator()

        # start threads to enqueue input minibatches for training
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # initialize all variables and ops
        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        # train
        for step in range(tf.app.flags.FLAGS.training_steps):

            train_op(sess, model, x_minibatch, y_minibatch, y_lengths_minibatch)

            # every 100 steps, evaluate model metrics and write summaries to disk
            if (step + 1) % 10 == 0 or step == tf.app.flags.FLAGS.training_steps - 1:
                eval_op(sess, model, x_minibatch, y_minibatch, y_lengths_minibatch, step, start_time)
                start_time = time.time()

        # when done, ask the threads to stop
        coord.request_stop()

        # wait for threads to finish
        coord.join(threads)

def train_op(sess, model, x_minibatch, y_minibatch, y_lengths_minibatch):
    x_values, y_values, y_lengths_values = sess.run([x_minibatch, y_minibatch, y_lengths_minibatch])

    # minimize loss
    sess.run([model.optimizer.minimize(model.loss), model.y_predicted_logits],
                                     feed_dict={model.x: x_values,
                                                model.y_actual: y_values,
                                                model.y_actual_lengths: y_lengths_values})