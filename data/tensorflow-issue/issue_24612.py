# tf.random.uniform((batch_size, feature_size), dtype=tf.int64) assumed sparse feature input for FMEstimator

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, feature_size=480000, cross_emb_size=4, negative_sampling_rate=1.0,
                 partitioner=None, optimizer=None):
        super().__init__()
        # Assumptions:
        # - feature_size: total number of features (480,000)
        # - cross_emb_size: embedding dimension for factorization machine cross terms (4)
        # - negative_sampling_rate: scalar parameter used in adjusted logits
        # - partitioner: tf variable partitioner or None
        # - optimizer: a tf.keras.optimizers.Optimizer instance or None
        
        self.feature_size = feature_size
        self.cross_emb_size = cross_emb_size
        self.negative_sampling_rate = negative_sampling_rate
        self.partitioner = partitioner
        
        # Initialize variables with partitioner if provided; use tf.Variable in tf2 idioms
        
        # linear bias (scalar)
        self.linear_bias = tf.Variable(
            initial_value=tf.random.normal([1], stddev=0.0001),
            dtype=tf.float32,
            trainable=True,
            name='linear_bias')
        
        # linear weights embedding matrix: shape [feature_size, 1]
        # Since tf.get_variable with partitioner is not idiomatic in tf2, we simulate
        self.linear_w = self._create_partitioned_variable(
            shape=[self.feature_size, 1],
            name='linear_w')
        
        # cross embedding weights matrix: shape [feature_size, cross_emb_size]
        self.cross_emb_w = self._create_partitioned_variable(
            shape=[self.feature_size, self.cross_emb_size],
            name='cross_emb_w')
        
        # Optimizer: if not provided, use Adam default
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam()
    
    def _create_partitioned_variable(self, shape, name):
        """
        Helper to create partitioned variable if partitioner is set,
        else normal variable.
        
        Since tf.Variable in TF2 does not support partitioner,
        we simulate by creating partitioned variables as list of variables.
        For simplicity, if no partitioner, create one variable.
        
        Notes:
        The original TF1 code used tf.get_variable with partitioner, resulting in tf.PartitionedVariable.
        To mimic that, we create a list of variables for partitioned parts or single var.
        """
        if self.partitioner is None:
            return tf.Variable(
                initial_value=tf.random.normal(shape, stddev=0.0001),
                dtype=tf.float32,
                trainable=True,
                name=name)
        else:
            # Simulate partitioned variable by dividing the first dimension
            # The partitioner can be of different type; we simplify assumption here:
            # If fixed_size_partitioner or min_max_variable_partitioner: 
            # partition first dimension into N parts
            partition_num = self.partitioner.num_shards if hasattr(self.partitioner, 'num_shards') else 1
            # divide shape[0] evenly into partition_num parts
            full_dim = shape[0]
            slice_size = (full_dim + partition_num - 1) // partition_num
            
            partitions = []
            for i in range(partition_num):
                start = i * slice_size
                end = min((i+1)*slice_size, full_dim)
                part_shape = [end - start] + list(shape[1:])
                if part_shape[0] <= 0:
                    # No slice if slice_size overflow, create empty variable to keep indices aligned
                    # Actually create zero size variable not possible, so skip
                    continue
                var_part = tf.Variable(
                    initial_value=tf.random.normal(part_shape, stddev=0.0001),
                    dtype=tf.float32,
                    trainable=True,
                    name='{}_part{}'.format(name, i))
                partitions.append(var_part)
            return partitions
    
    def _embedding_lookup_sparse(self, params, feature_sparse_tensor, combiner='sum'):
        """
        Args:
            params: either a tf.Variable or list of tf.Variables representing partitioned variables.
            feature_sparse_tensor: tf.sparse.SparseTensor with dtype int64, values are feature indices.
            combiner: 'sum', 'mean', or 'sqrtn'
            
        Returns:
            dense tensor [batch_size, embedding_dim] or [batch_size, 1] depending on params.
        """
        # If params is partitioned variables list, combine by embedding_lookup + concat
        if isinstance(params, list):
            # We need to perform embedding lookup on each partition and sum results
            # The key is to adjust lookups indices to slice
            splits = []
            offset = 0
            for part_var in params:
                splits.append(part_var.shape[0])
            splits = tf.constant(splits, dtype=tf.int64)
            
            # segment sparse_tensor indices based on which partition they belong to
            # feature_sparse_tensor.values are feature IDs
            
            values = feature_sparse_tensor.values  # [nnz]
            
            # To map values to partitions by range search on splits
            # Use tf.searchsorted to find which partition index each value belongs to
            # but tf.searchsorted is TF 2.x and above, so:
            # We'll manually do it by:
            # Iterate partitions and mask values
            
            # To avoid too complex logic, just loop in python and gather partial embeddings
            part_embeds = []
            batch_size = tf.shape(feature_sparse_tensor.dense_shape)[0]
            segment_ids = feature_sparse_tensor.indices[:, 0]
            
            # The input sparse tensor could be large, so running python for loop over partitions is suboptimal
            # For demonstration, do loop with masks, assuming small number of partitions (e.g. <16)
            
            for i, part_var in enumerate(params):
                start = sum(splits.numpy()[:i])
                end = start + splits.numpy()[i]
                
                # create mask of values within [start, end)
                mask = tf.logical_and(values >= start, values < end)
                masked_values = tf.boolean_mask(values, mask)
                # adjust indices relative to partition
                part_ids = masked_values - start
                
                # Get segment ids corresponding to masked values
                masked_segment_ids = tf.boolean_mask(segment_ids, mask)
                
                # Create sparse tensor for this partition
                # Indices: the same relative order of these nonzero elements
                sparse_part = tf.sparse.SparseTensor(
                    indices=tf.expand_dims(masked_segment_ids, 1), 
                    values=part_ids,
                    dense_shape=[batch_size])
                
                if part_var.shape.ndims == 2:
                    emb_dim = part_var.shape[1]
                else:
                    emb_dim = 1  # default
                
                # From sparse indices, do embedding_lookup_sparse:
                # Unfortunately tf.nn.embedding_lookup_sparse expects SparseTensor with shape [batch_size, ???]
                # but our sparse_part is rank 1, so we expand dims
                
                # We reshape sparse_part to have shape [batch_size, ?] by moving values to col 0
                # Note: TF expects sp_ids as SparseTensor with indices of shape [nnz, ndims].
                
                # sparse_part.indices shape [nnz, 1], need to add col dimension (column = 0)
                adjusted_indices = tf.concat([
                    sparse_part.indices,
                    tf.zeros_like(sparse_part.indices, dtype=tf.int64)
                ], axis=1)
                sp_ids = tf.sparse.SparseTensor(
                    indices=adjusted_indices,
                    values=sparse_part.values,
                    dense_shape=[batch_size, 1])
                
                lookup = tf.nn.embedding_lookup_sparse(
                    params=part_var,
                    sp_ids=sp_ids,
                    sp_weights=None,
                    combiner=combiner)  # shape [batch_size, emb_dim]
                part_embeds.append(lookup)
            
            # Sum across partitions
            if not part_embeds:
                # Handle corner case no partition created
                # Return zeros
                emb_dim_out = params[0].shape[1] if params else 1
                return tf.zeros([batch_size, emb_dim_out], dtype=tf.float32)
            sum_embeddings = tf.add_n(part_embeds)
            return sum_embeddings
        
        else:
            # params is a single variable
            return tf.nn.embedding_lookup_sparse(params=params,
                                                sp_ids=feature_sparse_tensor,
                                                sp_weights=None,
                                                combiner=combiner)
    
    def __embedding_lookup_square_sparse(self, params, sp_ids, combiner="mod"):
        """
        Compute sum of element-wise squares of embeddings selected by sparse IDs.
        Equivalent to "sum of squared features" for FM second order.
        
        Args:
            params: variable or partitioned variables (list)
            sp_ids: tf.sparse.SparseTensor int64
        
        Returns:
            tensor of shape [batch_size, embedding_dim]
        """
        # Get embeddings for all ids
        embeddings = self._embedding_lookup_sparse(params, sp_ids, combiner='sum')  # shape [batch_size, emb_dim]
        embeddings_squared = tf.square(embeddings)
        
        # Get squared embeddings summed by segment: sum(e_i^2)
        # For segment_ids, use sp_ids.indices[:,0]
        segment_ids = sp_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)
        
        # Unique ids and idx mapping are normally used in TF1 code; here approximate by direct sparse ops:
        # Instead of unique and segment_sum on embeddings squared, 
        # we do segment sum on squared individual embeddings obtained by separate embedding_lookup on squared params.
        
        # Compute summed squared embeddings by sparse segment sum (TF2 compatible)
        # To do this, first get embeddings of squared params per feature and weights per instance
        # We replicate the original trick: sum(e_i^2) by summing squared individual embeddings per feature indices.
        
        # Embedding squared params per feature:
        # For partitioned variable: gather squared params and lookup sparse
        
        # To simplify: Take square of params and do sparse lookup sum
        # If params is a list of variables, square them individually and then lookup sparse sum and sum parts
        
        def square_params_lookup_sum(params, sp_ids):
            if isinstance(params, list):
                total_sum = 0
                for part_var in params:
                    part_var_sq = tf.square(part_var)
                    total_sum += tf.nn.embedding_lookup_sparse(part_var_sq, sp_ids, sp_weights=None, combiner='sum')
                return total_sum
            else:
                return tf.nn.embedding_lookup_sparse(tf.square(params), sp_ids, sp_weights=None, combiner='sum')
        
        summed_squared_cross_emb = square_params_lookup_sum(params, sp_ids)
        
        return embeddings_squared, summed_squared_cross_emb
    
    @tf.function
    def call(self, inputs, training=False):
        # inputs is dict with key 'featureID' containing a tf.sparse.SparseTensor of int64 feature indices
        feature_sparse = inputs['featureID']
        
        # -- linear term --
        # lookup linear weights embedding and sum
        logits_wide = self._embedding_lookup_sparse(self.linear_w, feature_sparse, combiner='sum')  # [batch_size, 1]
        logits_linear = self.linear_bias + logits_wide  # broadcast bias
        
        # -- cross term for FM --
        summed_cross_emb = self._embedding_lookup_sparse(self.cross_emb_w, feature_sparse, combiner='sum')  # [batch_size, cross_emb_size]
        
        squared_summed_cross_emb = tf.square(summed_cross_emb)
        
        # summed squared embeddings
        _, summed_squared_cross_emb = self.__embedding_lookup_square_sparse(self.cross_emb_w, feature_sparse)
        
        # factorization machines cross logits component
        logits_cross = 0.5 * tf.reduce_sum(squared_summed_cross_emb - summed_squared_cross_emb, axis=1, keepdims=True)  # [batch_size, 1]
        
        # final logits
        # The original code commented out adding cross logits in training: logits = logits_linear
        # We'll output the sum of both linear and cross terms, as in FM models
        logits = logits_linear + logits_cross
        
        # logits adjusted with negative sampling rate log
        logits_adjusted = logits + tf.math.log(self.negative_sampling_rate)
        
        # for compatibility return dictionary output similar to EstimatorSpec predictions
        outputs = {
            'logits': logits,
            'logits_adjusted': logits_adjusted,
            'probabilities': tf.nn.sigmoid(logits_adjusted)
        }
        
        return outputs
    
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            preds = self.call(inputs, training=True)
            logits = preds['logits']
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(labels, tf.float32),
                    logits=logits))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

def my_model_function():
    # Create an instance of MyModel with typical parameters and default no partitioner
    # For demonstration, no partitioner and default Adam optimizer
    # User can provide partitioner if necessary, e.g. tf.fixed_size_partitioner(num_shards=4)
    
    # Example partitioner simulation class with num_shards attribute for _create_partitioned_variable
    class FixedSizePartitioner:
        def __init__(self, num_shards):
            self.num_shards = num_shards
    
    # Provide simple partitioner with 4 shards as example (can be None)
    partitioner = None  # Or FixedSizePartitioner(num_shards=4)
    
    model = MyModel(
        feature_size=480000,
        cross_emb_size=4,
        negative_sampling_rate=1.0,
        partitioner=partitioner,
        optimizer=tf.keras.optimizers.Adam())
    return model

def GetInput():
    # Returns example input dictionary matching expected input: sparse tensor with features 'featureID'
    # Batch size arbitrary, e.g. batch_size=5
    batch_size = 5
    # Randomly generate sparse features indices (feature IDs) for demonstration
    import numpy as np
    
    # For simplicity: generate random sparse indices and values
    # Indices shape [nnz, 2]: [batch_index, feature_index]
    nnz = 10  # number of nonzero total entries in sparse tensor
    
    batch_indices = np.random.randint(0, batch_size, size=(nnz,))
    feat_indices = np.random.randint(0, 480000, size=(nnz,))
    indices = np.stack([batch_indices, feat_indices], axis=1)
    
    values = feat_indices.astype(np.int64)  # ids same as feature indices
    
    dense_shape = np.array([batch_size, 480000], dtype=np.int64)
    
    # Create sparse tensor featureID with indices shape [nnz, 2], values int64 feature IDs
    sp_ids = tf.sparse.SparseTensor(
        indices=tf.constant(indices, dtype=tf.int64),
        values=tf.constant(values, dtype=tf.int64),
        dense_shape=tf.constant(dense_shape, dtype=tf.int64))
    
    return {'featureID': sp_ids}

