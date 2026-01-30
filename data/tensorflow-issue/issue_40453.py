import tensorflow as tf

class LossAccObserver:
    def __init__(self):
        self.loss = metrics.SparseCategoricalCrossentropy()
        self.acc = metrics.SparseCategoricalAccuracy()
    def reset(self):
        self.loss.reset_states()
        self.acc.reset_states()
    def update(self, y, y_hat):
        self.loss.update_state(y, y_hat)
        self.acc.update_state(y, y_hat)

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

@tf.function
def train_epoch(model, strategy, batch_size, dataset, obs):
    def step_fn(x, y):
        y_hat = compute_and_apply_gradients(model, x, y)
        obs.update(y, y_hat)
    for x, y in dataset:
        strategy.run(step_fn, args = (x, y))

@tf.function
def evaluate_epoch(model, strategy, dataset, obs):
    def step_fn(x, y):
        y_hat = model(x, training = False)
        obs.update(y, y_hat)
    for x, y in dataset:
        strategy.run(step_fn, args = (x, y))

def manual_training(model, strategy, train, valid, batch_size, epochs):
    with strategy.scope():
        train_obs = LossAccObserver()
        valid_obs = LossAccObserver()
    ...
    for i in range(epochs):
        ...
        train_epoch(model, strategy, batch_size, train, train_obs)
        evaluate_epoch(model, strategy, valid, valid_obs)
        ...

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

def automatic_training(model, train, valid, batch_size, epochs):
    ...
    model.fit(x = train, validation_data = valid,
              epochs = epochs,
              verbose = 2)