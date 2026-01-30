from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import functools
from typing import Optional


class TokenWeightingLayer(tf.keras.layers.Layer):
    def __init__(self, mask_constant=-50000, **kwargs):
        super(TokenWeightingLayer, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self._mask_constant = mask_constant

    def call(self, inputs, mask, **kwargs):
        activation = tf.squeeze(self.fc(tf.tanh(inputs)), axis=-1)
        masked_activation = tf.where(mask, activation,
                                     (tf.ones_like(activation) * self._mask_constant))  # TODO: fix dtype
        alpha = self.softmax(masked_activation)
        return alpha


class RNN_Attention_Clf(tf.keras.Model):
    def __init__(self,
                 word_embedding_dim: int,
                 pos1_embedding_dim: int,
                 pos2_embedding_dim: int,
                 vocab_size: int,
                 pos_embedding_num: int,
                 rnn_dim: int,
                 dropout_rate: float = 0.,
                 pre_trained_word_embedding: np.ndarray = None,
                 fine_tune_word_embedding: bool = False,
                 variational_recurrent: bool = False,
                 num_cls: Optional[int] = None,
                 **kwargs):
        super(RNN_Attention_Clf, self).__init__()
        self._word_embedding_dim = word_embedding_dim
        self._entity_1_embedding_dim = pos1_embedding_dim
        self._entity_2_embedding_dim = pos2_embedding_dim
        self.cell_type = tf.keras.layers.GRUCell
        self._vocab_size = vocab_size  # int, without unk and pad
        self._pos_embedding_num = pos_embedding_num
        self._pre_trained_word_embedding = pre_trained_word_embedding  # numpy.ndarray object
        self._fine_tune_word_embedding = (self._pre_trained_word_embedding is None) or kwargs.get(
            "fine_tune_word_embedding", False)
        self._variational_recurrent = variational_recurrent
        self._rnn_dim = rnn_dim  # int
        self._num_cls = num_cls
        self._dropout_rate = dropout_rate
        if pre_trained_word_embedding is None:
            self.token_embedding = tf.keras.layers.Embedding(self._vocab_size, self._word_embedding_dim, mask_zero=True)
        else:
            raise NotImplementedError("WIP")
        self._pos_embedding_1 = tf.keras.layers.Embedding(self._pos_embedding_num, self._word_embedding_dim,
                                                          mask_zero=True)
        self._pos_embedding_2 = self._pos_embedding_1
        cell = tf.keras.layers.GRUCell(rnn_dim)
        rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
        self.rnn_encoder = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")
        self.token_weighting_layer = TokenWeightingLayer()
        self.fc_layer = tf.keras.layers.Dense(num_cls, use_bias=True)

    def call(self, token_ids, pos_ids_1, pos_ids_2, sequence_length, training=None, mask=None):
        token_embedding = self.token_embedding(token_ids)
        # token_dropout
        pos_embedding_1 = self._pos_embedding_1(pos_ids_1)
        pos_embedding_2 = self._pos_embedding_2(pos_ids_2)
        encoded_input = tf.concat([token_embedding, pos_embedding_1, pos_embedding_2], axis=-1, )
        # Compute mask
        mask = tf.sequence_mask(sequence_length, maxlen=token_ids.get_shape()[1])
        # BiRNN
        encoder_outputs, last_state_fw, last_state_bw = self.rnn_encoder(encoded_input, mask=mask)

        # The original authors of this network call this "attention", well...
        alpha = self.token_weighting_layer(encoder_outputs, mask)
        weighted_outputs = tf.squeeze(tf.matmul(tf.expand_dims(alpha, axis=1), encoder_outputs), axis=1)
        out = self.fc_layer(weighted_outputs)
        return out, alpha

    def model_fn(self, features, labels, mode, params):
        # dropout_rate = params["dropout_rate"]
        if labels is not None:
            sparse_label_ids = labels
        # TODO: Set configure RNN dropout for RNN input, state, output
        input_text, inputs, pos1_inputs, pos2_inputs, sequence_length = features
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not training:
            self.trainable = False  # Do I have to?
        logits, alpha = self(inputs, pos1_inputs, pos2_inputs, sequence_length)
        print(self.summary())

        top1_prediction = tf.argmax(logits, axis=-1, name="top1_cls")

        confidence = tf.nn.softmax(logits, axis=-1, name="confidence")
        top1_confidence = tf.reduce_max(confidence, axis=-1)
        if mode != tf.estimator.ModeKeys.PREDICT:
            def loss_fn():  # For training stage only.
                logits, alpha = self(inputs, pos1_inputs, pos2_inputs, sequence_length,
                                     training=True)  # Apply dropout, batch-normalization etc. when calculating loss.
                loss = tf.keras.losses.sparse_categorical_crossentropy(sparse_label_ids, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
                return loss

            global_step = tf.compat.v1.train.get_or_create_global_step()
            y_true = tf.one_hot(sparse_label_ids, depth=self._num_cls)
            metrics = {}
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=sparse_label_ids, y_pred=confidence,
                                                                   from_logits=False)
            loss = tf.reduce_mean(loss)
            acc = tf.metrics.SparseTopKCategoricalAccuracy(k=1)
            acc.update_state(y_true=sparse_label_ids, y_pred=confidence)
            metrics["acc"] = acc
            for cls_id in range(self._num_cls):
                p = tf.metrics.Precision(top_k=1, class_id=cls_id)
                p.update_state(y_true=y_true, y_pred=confidence)
                metrics["precision_cls_{}".format(cls_id)] = p
                r = tf.metrics.Recall(top_k=1, class_id=cls_id)
                r.update_state(y_true=y_true, y_pred=confidence)
                metrics["recall_cls_{}".format(cls_id)] = r

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "cls_id": top1_prediction,
                "confidence": confidence,
                "att_weight": alpha,
                "input_text": input_text
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            init_lr = params["init_lr"]
            decay_step = params["decay_step"]
            decay_rate = params["decay_rate"]
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, decay_steps=decay_step,
                                                                           decay_rate=decay_rate)

            if params["optimizer"] == 'Adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif params["optimizer"] == 'Adadelta':
                optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
            elif params["optimizer"] == 'Adagrad':
                optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
            elif params["optimizer"] == 'RMSProp':
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif params["optimizer"] == 'Momentum':
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            elif params["optimizer"] == 'SGD':
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            trainable_variables = self.trainable_variables
            optimizer.iterations = global_step
            tf.summary.scalar("learning_rate", learning_rate(optimizer.iterations), global_step)
            clip_val = params.get("clip_value")
            if clip_val is not None:
                # Tampering gradients manually instead of setting optimizer.clipvalue
                with tf.GradientTape() as tape:
                    loss_val = loss_fn()
                gradients = tape.gradient(loss_val, trainable_variables)
                gradients = [(tf.clip_by_value(grad, -clip_val, clip_val)) if grad is not None else grad for grad in
                             gradients]
                assert len(trainable_variables) == len(gradients)
                for var, grad in zip(trainable_variables, gradients):
                    print("****variable:{} Gradient: {}".format(var, grad))
                train_op = optimizer.apply_gradients(zip(gradients, trainable_variables))
            else:
                train_op = optimizer.minimize(loss_fn, var_list=trainable_variables)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            raise ValueError("Unknown mode: {}".format(mode))



def test_non_estimator_approach():
    nn = RNN_Attention_Clf(word_embedding_dim=64, pos1_embedding_dim=5, pos2_embedding_dim=5, pos_embedding_num=6,
                           vocab_size=100, rnn_dim=20, num_cls=10)
    inputs = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]])
    pos_1 = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])
    pos_2 = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]])
    seq_length = tf.constant([5, 4, 8])
    sparse_label_ids = tf.constant([0, 1, 2])
    def loss_fn():  # For training stage only.
        logits, alpha = nn(inputs, pos_1, pos_1, seq_length,
                           training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(sparse_label_ids, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
        return loss

    out = nn(inputs, pos_1, pos_2, seq_length)
    print(nn.summary())  # Variables in `directional layers are trainable
    clip_val = 5
    if clip_val is not None:
        with tf.GradientTape() as tape:
            loss_val = loss_fn()

    grads = tape.gradient(loss_val, nn.trainable_variables)
    assert len(nn.trainable_variables) == len(grads)
    for var, grad in zip(nn.trainable_variables, grads):
        print("****variable:{} Gradient: {}".format(var, grad))


def test_estimator_approach():
    nn = RNN_Attention_Clf(word_embedding_dim=64, pos1_embedding_dim=5, pos2_embedding_dim=5, pos_embedding_num=6,
                           vocab_size=100, rnn_dim=20, num_cls=10)
    params = {"clip_value": 5, "init_lr": 0.1, "clip_val": 5, "decay_step": 1, "decay_rate": 0.99, "optimizer": "Adam"}
    output_dir = "./test_output"
    estimator = tf.estimator.Estimator(nn.model_fn,
                                       config=tf.estimator.RunConfig(save_checkpoints_secs=20, save_summary_steps=1),
                                       model_dir=output_dir,
                                       params=params)
    def input_fn():
        def gen():
            dummy_text = ["text1", "text2", "text3"]
            dummy_labels = [0, 1, 2]
            all_inputs = [[1, 2, 3, 4, 5, 0, 0, 0], [1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]]
            all_pos1 = [[1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]]
            all_pos2 = [[1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0]]
            seq_length = [5, 4, 8]
            for text, inputs, pos1, pos2, sequence_length, y in zip(dummy_text, all_inputs, all_pos1, all_pos2, seq_length, dummy_labels):
                out = ((text, inputs, pos1, pos2, sequence_length), y)
                yield out
        d = tf.data.Dataset.from_generator(gen,
                                           output_types=((tf.string, tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
                                           output_shapes=(((), (None,), (None,), (None,), ()), ()))
        d = d.batch(batch_size=3)
        return d
    estimator.train(input_fn=input_fn, max_steps=100)


if __name__ == "__main__":
    # To replicate the issue, clear `./test_output` before re-running

    test_non_estimator_approach()  # No problem with this approach
    test_estimator_approach()  # No gradient computed for weights in GRUCell