from torchrl.envs import GymEnv
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SliceSampler

sampler = SliceSampler(slice_len=10, end_key=("next", "done"), compile=True)
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(100, device="cpu"),
    sampler=sampler,
    batch_size=10,
    prefetch=2,
)

env = GymEnv("CartPole-v1")
rollout_data = env.rollout(max_steps=100, break_when_any_done=False)
replay_buffer.extend(rollout_data)

sample = replay_buffer.sample()