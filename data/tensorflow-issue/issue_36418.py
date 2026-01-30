import tensorflow as tf
import numpy as np

from tf_agents import specs
from tf_agents.replay_buffers import episodic_replay_buffer


class EpisodicReplayBuffer:
    def __init__(self, example_len, buffer_size):
        self.example_len = example_len
        self.buffer_size = buffer_size

        self.episodes_collected = 0

    def initialize(self, episode):
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
                print("Unknown dtype: {}".format(dtype))
                quit()
            shape = list(value.shape)[2:]
            spec[key] = specs.TensorSpec(shape, dtype, key)

        self.replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
            spec,
            capacity=self.buffer_size,
            begin_episode_fn=lambda _: False,
            end_episode_fn=lambda _: True,
        )
        # self.add_episode(episode)

    def add_episode(self, episode):
        print("Episode collected: {}\n".format(self.episodes_collected))
        self.episodes_collected += 1
        episode_id = self.replay_buffer.create_episode_ids(num_episodes=0)
        episode = tf.nest.map_structure(lambda v: v[0], episode)
        self.replay_buffer.add_sequence(episode, episode_id)

    def as_dataset(self, batch_size):
        return self.replay_buffer._as_dataset(
            num_steps=self.example_len, sample_batch_size=batch_size
        )


def main():
    strategy = tf.distribute.MirroredStrategy()
    replay_buffer = EpisodicReplayBuffer(4, 4)
    episode = {}
    for key in ["tensor_1", "tensor_2", "tensor_3"]:
        value = np.ones(shape=[1, 4, 4, 4], dtype=np.float32)
        episode[key] = value

    replay_buffer.initialize(episode)
    with strategy.scope():
        replay_buffer.add_episode(episode)
        replay_buffer.add_episode(episode)
        replay_buffer.add_episode(episode)
        dataset = replay_buffer.as_dataset(4)
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(distributed_dataset)
        data = next(iterator)
        print(data)

if __name__ == "__main__":
    main()