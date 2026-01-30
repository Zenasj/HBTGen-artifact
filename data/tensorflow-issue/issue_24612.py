import math
import tensorflow as tf

class FMEstimator:
	def __init__(self, model_dir, config=None, params=None, optimizer=None, partitioner=None):
		self.model_dir = model_dir
		self.config = config
		self.params = params
		self.optimizer = optimizer
		self.partitioner = partitioner

	def __embedding_lookup_square_sparse(self, params, sp_ids, partition_strategy="mod", name=None, max_norm=None):
		if isinstance(params, variables.PartitionedVariable):
			params = list(params)  # Iterate to get the underlying Variables.
		if not isinstance(params, list):
			params = [params]

		with ops.name_scope(name, "embedding_lookup_square_sparse",
		                    params + [sp_ids]) as name:
			segment_ids = sp_ids.indices[:, 0]
			if segment_ids.dtype != dtypes.int32:
				segment_ids = math_ops.cast(segment_ids, dtypes.int32)

			ids = sp_ids.values
			ids, idx = array_ops.unique(ids)

			embeddings = tf.nn.embedding_lookup(params, ids, partition_strategy=partition_strategy, max_norm=max_norm)

			embeddings = tf.square(embeddings)

			assert idx is not None

			embeddings = math_ops.sparse_segment_sum(embeddings, idx, segment_ids, name=name)

			return embeddings

	def get_model_fn(self):
		def custom_model_fn(features, labels, mode, params):
			linear_bias = tf.get_variable(name='linear_bias',
			                              shape=[1],
			                              dtype=tf.float32,
			                              initializer=tf.random_normal_initializer(stddev=0.0001))

			linear_w = tf.get_variable(name='linear_w',
			                           shape=[params['feature_size'], 1],
			                           dtype=tf.float32,
			                           initializer=tf.random_normal_initializer(stddev=0.0001),
			                           partitioner=self.partitioner)

			# wx
			# size: [batch_size, 1]
			logits_wide = tf.nn.embedding_lookup_sparse(params=linear_w,
			                                            sp_ids=features['featureID'],
			                                            sp_weights=None,
			                                            combiner='sum')
			# wx + b
			logits_linear = linear_bias + logits_wide

			cross_emb_w = tf.get_variable(name='cross_emb_w',
			                              shape=[params['feature_size'], params['cross_emb_size']],
			                              dtype=tf.float32,
                                                      partitioner=self.partitioner,
			                              initializer=tf.random_normal_initializer(stddev=0.0001))

			# (a1,b1,c1) (a2,b2,c2) -> (a1+a2, b1+b2, c1+c2)
			# size: [batch_size, cross_emb_size]
			summed_cross_emb = tf.nn.embedding_lookup_sparse(params=cross_emb_w,
			                                                 sp_ids=features['featureID'],
			                                                 sp_weights=None,
			                                                 combiner='sum')
			# ((a1+a2)^2, (b1+b2)^2, (c1+c2)^2)
			# size: [batch_size, cross_emb_size]
			squared_summed_cross_emb = tf.square(summed_cross_emb)

			# (a1^2, b1^2, c1^2) (a2^2, b2^2, c2^2) -> (a1^2+a2^2, b1^2+b2^2, c1^2+c2^2)
			# size: [batch_size, cross_emb_size]
			summed_squared_cross_emb = self.__embedding_lookup_square_sparse(params=cross_emb_w,
			                                                                 sp_ids=features['featureID'])

			# a1a2+b1b2+c1c2
			# size: [batch_size, 1]
			logits_cross = tf.reduce_sum(0.5 * tf.subtract(squared_summed_cross_emb, summed_squared_cross_emb), 1, keepdims=True)

			# logits = tf.add(logits_linear, logits_cross)
			logits = logits_linear
			logits_adjusted = logits + tf.math.log(params['negative_sampling_rate'])

			if mode == tf.estimator.ModeKeys.PREDICT:
				predictions = {
					'probabilities': tf.nn.sigmoid(logits_adjusted),
					'logits': logits,
					'logits_adjusted': logits_adjusted
				}

				return tf.estimator.EstimatorSpec(mode, predictions=predictions)

			else:
				loss = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32),
					                                        logits=logits))

				if mode == tf.estimator.ModeKeys.EVAL:
					auc = tf.metrics.auc(
						labels=labels,
						predictions=1 / (1 + tf.math.exp(-logits_adjusted)),
						num_thresholds=400,
						curve='ROC',
						summation_method='careful_interpolation')
					logloss = tf.metrics.mean(tf.nn.sigmoid_cross_entropy_with_logits(
						labels=tf.cast(labels, dtype=tf.float32),
						logits=logits_adjusted))
					tf.summary.scalar('True_AUC', auc)
					tf.summary.scalar('True_Logloss', logloss)
					metrics = {
						'True_AUC': auc,
						'True_Logloss': logloss
					}

					predictions = {
						'probabilities': tf.nn.sigmoid(logits_adjusted),
						'logits': logits,
						'logits_adjusted': logits_adjusted
					}

					return tf.estimator.EstimatorSpec(mode, loss=loss, predictions=predictions,
					                                  eval_metric_ops=metrics)

				elif mode == tf.estimator.ModeKeys.TRAIN:
					train_op = self.optimizer.minimize(loss, global_step=tf.train.get_global_step())

					return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

		return custom_model_fn

	def get_estimator(self):
		return tf.estimator.Estimator(model_fn=self.get_model_fn(),
		                              model_dir=self.model_dir,
		                              config=self.config,
		                              params=self.params)