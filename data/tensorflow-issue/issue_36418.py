# tf.random.uniform((B, 4, 4, 4), dtype=tf.float32)
import tensorflow as tf
import numpy as np
from tf_agents import specs
from tf_agents.replay_buffers import episodic_replay_buffer

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We encapsulate the EpisodicReplayBuffer logic inside this model as a submodule-like member.
        # EpisodicReplayBuffer is not a keras layer, but for the purpose of this task,
        # we integrate its add_episode and dataset retrieval functions here.
        self.example_len = 4
        self.buffer_size = 4
        self.episodes_collected = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.replay_buffer = None
        self.spec = None

    def initialize_buffer(self, episode):
        # Infer specs from example episode dictionary
        spec = dict()
        for key, value in episode.items():
            dtype = str(value.dtype)
            if dtype == "int32":
                dtype = tf.int32
            elif dtype == "int64":
                dtype = tf.int64
            elif dtype == "float32":
                dtype = tf.float32
            elif dtype == "float64":
                dtype = tf.float64
            elif dtype == "bool":
                dtype = tf.bool
            else:
                raise ValueError(f"Unknown dtype encountered: {dtype}")
            # Shape excludes batch dims: original shape is [1, 4, 4, 4]
            # The example took shape from index 2 onward (i.e. [4, 4, 4])
            shape = list(value.shape)[2:]  
            spec[key] = specs.TensorSpec(shape, dtype, key)
        self.spec = spec
        self.replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
            spec,
            capacity=self.buffer_size,
            begin_episode_fn=lambda _: False,
            end_episode_fn=lambda _: True,
        )

    @tf.function
    def add_episode(self, episode):
        tf.print("Episode collected:", self.episodes_collected)
        self.episodes_collected.assign_add(1)
        episode_id = self.replay_buffer.create_episode_ids(num_episodes=0)
        # episode is expected as dict of tensors shaped [1, 4, 4, 4], extract first dimension ([0, ...])
        episode_trimmed = tf.nest.map_structure(lambda v: v[0], episode)
        self.replay_buffer.add_sequence(episode_trimmed, episode_id)

    def as_dataset(self, batch_size):
        return self.replay_buffer._as_dataset(
            num_steps=self.example_len, sample_batch_size=batch_size
        )

    def call(self, inputs, training=False):
        # The forward pass returns the data from the dataset iterator.
        # Because tf-agents EpisodicReplayBuffer is not inherently a keras layer,
        # and we have no trainable transformations here,
        # we simulate one "step" of dataset retrieval.
        strategy = tf.distribute.get_strategy()
        dataset = self.as_dataset(batch_size=inputs.shape[0])
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(distributed_dataset)
        data = next(iterator)
        return data

def my_model_function():
    # Create a MyModel instance and initialize its replay buffer with dummy data to make it ready for calls.
    model = MyModel()
    # Create dummy episode to initialize buffer specs
    episode = {}
    for key in ["tensor_1", "tensor_2", "tensor_3"]:
        # Shape: [1, 4, 4, 4], dtype float32, matching example in issue
        episode[key] = tf.ones(shape=[1, 4, 4, 4], dtype=tf.float32)
    model.initialize_buffer(episode)
    # Add a few episodes to avoid empty dataset errors
    model.add_episode(episode)
    model.add_episode(episode)
    model.add_episode(episode)
    return model

def GetInput():
    # According to the example, the input is an episode dictionary with 3 keys,
    # each a batch of shape [B, 4, 4, 4], dtype float32.
    B = 4
    input_dict = {}
    for key in ["tensor_1", "tensor_2", "tensor_3"]:
        input_dict[key] = tf.random.uniform((B, 4, 4, 4), dtype=tf.float32)
    return input_dict

