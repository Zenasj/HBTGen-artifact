import numpy as np
import random
import tensorflow as tf

from models import *
from conftest import DDPGAgent
import matplotlib as plt
import pytest
import time

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# setting for hidden layers
Layer1 = 400
Layer2 = 300


class MecTer(object):
    """
    MEC terminal parent class
    """

    def __init__(self, user_config, train_config):
        self.rate = user_config['rate']
        self.dis = user_config['dis']
        self.id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.data_buf_size = user_config['data_buf_size']
        self.t_factor = user_config['t_factor']
        self.penalty = user_config['penalty']

        self.sigma2 = train_config['sigma2']
        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        if 'model' not in user_config:
            self.channelModel = MarkovModel(self.dis, seed=train_config['random_seed'])
        else:
            n_t = 1
            n_r = user_config['num_r']
            self.channelModel = ARModel(self.dis, n_t, n_r, seed=train_config['random_seed'])

        self.DataBuf = 0
        self.Channel = self.channelModel.getCh()
        self.SNR = 0
        self.Power = np.zeros(self.action_dim)
        self.Reward = 0
        self.State = []

        # some pre-defined parameters
        self.k = 1e-27
        self.t = 0.001
        self.L = 500

    def localProc(self, p):
        return np.power(p / self.k, 1.0 / 3.0) * self.t / self.L / 1000

    def localProcRev(self, b):
        return np.power(b * 1000 * self.L / self.t, 3.0) * self.k

    def offloadRev(self, b):
        return (np.power(2.0, b) - 1) * self.sigma2 / np.power(np.linalg.norm(self.Channel), 2)

    def offloadRev2(self, b):
        return self.action_bound if self.SNR <= 1e-12 else (np.power(2.0, b) - 1) / self.SNR

    def getCh(self):
        return self.Channel

    def setSNR(self, snr):
        self.SNR = snr
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel), 2) / self.sigma2
        self.State = np.array([self.DataBuf, snr, channel_gain])

    def sampleData(self):
        data_t = np.log2(1 + self.Power[0] * self.SNR)
        data_p = self.localProc(self.Power[1])
        over_power = 0

        self.DataBuf -= data_t + data_p
        if self.DataBuf < 0:
            over_power = self.Power[1] - self.localProcRev(np.fmax(0, self.DataBuf + data_p))
            self.DataBuf = 0

        data_r = np.random.poisson(self.rate)
        self.DataBuf += data_r
        return data_t, data_p, data_r, over_power

    def sampleCh(self):
        # self.Channel = self.channelModel.sampleCh()

        # Calculate channel gain using channel quantization
        raw_channel_gain = np.linalg.norm(self.channelModel.sampleCh())
        min_val = np.min(self.Channel)
        max_val = np.max(self.Channel)

        # Quantize the channel gain into 10 levels
        quantized_channel_gain = min_val + (max_val - min_val) * (raw_channel_gain - min_val) / (max_val - min_val)
        quantized_channel_gain = np.clip(quantized_channel_gain, min_val, max_val)
        self.Channel = quantized_channel_gain

        return self.Channel

    def reset(self, rate, seqCount):
        self.rate = rate
        self.DataBuf = np.random.randint(0, self.data_buf_size - 1) / 2.0
        self.sampleCh()

        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True

        return self.DataBuf


class MecTermRL(MecTer):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, user_config, train_config):
        MecTer.__init__(self, user_config, train_config)
        self.agent = DDPGAgent(user_config, train_config)

        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False

    def feedback(self, snr, done):
        isOverflow = 0
        self.SNR = snr

        # update the data buffer
        [data_t, data_p, data_r, over_power] = self.sampleData()

        # get the reward for the current slot
        self.Reward = -self.t_factor * np.sum(self.Power) * 10 - (1 - self.t_factor) * self.DataBuf

        # estimate the channel for next slot
        self.sampleCh()

        # update the actor and critic network
        channel_gain = np.power(np.linalg.norm(self.Channel), 2) / self.sigma2
        next_state = np.array([self.DataBuf, snr, channel_gain])

        self.agent.update(self.State, self.Power, self.Reward, done, next_state, self.isUpdateActor)

        # update system state
        self.State = next_state
        # return the reward in this slot
        sum_power = np.sum(self.Power) - over_power
        return self.Reward, sum_power, over_power, data_t, data_p, data_r, self.DataBuf, channel_gain, isOverflow

    def predict(self, isRandom):
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        self.Power = np.fmax(0, np.fmin(self.action_bound, power))

        return self.Power, noise


class MecSvrEnv(object):
    """
    Simulation environment
    """

    def __init__(self, user_list, num_att, sigma2, max_len):
        self.user_list = user_list
        self.num_user = len(user_list)
        self.num_att = num_att
        self.sigma2 = sigma2
        self.count = 0
        self.seqCount = 0
        self.max_len = max_len

        # specially designed for Greedy agent training

    #         self.data_set = []

    def init_target_network(self):
        for user in self.user_list:
            user.critic.init_target_network(path='data_set_OGD.npz')

    def plot_channel_gains_histogram(self):
        # Get the channel gains for all users
        channel_gains = [np.abs(user.getCh()) for user in self.user_list]

        # Flatten the channel gains to a 1D array
        flat_channel_gains = np.concatenate(channel_gains)

        # plot a histogram for the channel gains
        plt.hist(np.abs(flat_channel_gains), bins=20, edgecolor='black')
        plt.title("Channel Gains Histogram")
        plt.xlabel("Channel Gain Magnitude")
        plt.ylabel("Frequency")
        plt.show()

    def step_transmit(self, isRandom=True):
        # get the channel vectors
        channels = np.transpose([user.getCh() for user in self.user_list])
        # get the transmit powers
        powers = []
        noises = []

        for i in range(self.num_user):
            p, n = self.user_list[i].predict(isRandom)
            powers.append(p.copy())
            noises.append(n.copy())
        # compute the snr for each user

        powers = np.array(powers)
        noises = np.array(noises)
        snr_list = self.compute_snr(channels, powers[:, 0])

        rewards = np.zeros(self.num_user)
        powers = np.zeros(self.num_user)
        over_powers = np.zeros(self.num_user)
        data_ts = np.zeros(self.num_user)
        data_ps = np.zeros(self.num_user)
        data_rs = np.zeros(self.num_user)
        data_buf_sizes = np.zeros(self.num_user)
        next_channels = np.zeros(self.num_user)
        isOverflows = np.zeros(self.num_user)

        self.count += 1
        # feedback the snr to each user
        for i in range(self.num_user):
            [rewards[i], powers[i], over_powers[i], data_ts[i], data_ps[i], data_rs[i], data_buf_sizes[i],
             next_channels[i], isOverflows[i]] = self.user_list[i].feedback(snr_list[i], self.count >= self.max_len)

        return rewards, self.count >= self.max_len, powers, over_powers, noises, data_ts, data_ps, data_rs, data_buf_sizes, next_channels, isOverflows

    def compute_snr(self, channels, powers):
        # FDD - Computing SNR
        H_inv = np.linalg.pinv(channels)
        total_signal_power = np.power(np.linalg.norm(channels, axis=1), 2)
        noise = np.power(np.linalg.norm(H_inv, axis=1), 2) * self.sigma2
        snr_list = total_signal_power / noise

        return snr_list

    def reset(self, isTrain=True):
        self.count = 0

        if isTrain:
            init_data_buf_size = [user.reset(user.rate, self.seqCount) for user in self.user_list]
            # get the channel vectors
            channels = np.transpose([user.getCh() for user in self.user_list])
            # get the transmit powers to start
            powers = [np.random.uniform(0, user.action_bound) for user in self.user_list]
            # compute the snr for each user
            snr_list = self.compute_snr(channels, powers)
        else:
            init_data_buf_size = [0 for user in self.user_list]
            snr_list = [0 for user in self.user_list]

        for i in range(self.num_user):
            self.user_list[i].setSNR(snr_list[i])

        self.seqCount += 1
        return init_data_buf_size


# Create the environment
# def env():
#     envi = MecSvrEnv(user_list, NUM_R, SIGMA2, MAX_EPISODE_LEN)
#     return envi

# env = MecSvrEnv(user_list, NUM_R, SIGMA2, MAX_EPISODE_LEN)
# env.init_target_network()

train_config = {
    'sigma2': 0.01,
    'minibatch_size': 64,
    'actor_lr': 0.0001,
    'tau': 0.001,
    'critic_lr': 0.001,
    'gamma': 0.99,
    'buffer_size': 250000,
    'random_seed': int(time.perf_counter() * 1000 % 1000),
    'noise_sigma': 0.12
}

# Define user_list_info with user information
user_list_info = [
    {'state_dim': 3,
     'action_dim': 1,
     'id': '1',
     'action_bound': 1,
     'model': 'AR',
     'num_r': 4,
     'rate': 3.0,
     'dis': 100,
     'data_buf_size': 100,
     't_factor': 1.0,
     'penalty': 1000, }
]

# sess = tf.compat.v1.Session()
# Create instances of the User class from the dictionary in user_list
user_list = [
    MecTermRL(user_config=user_info, train_config=train_config)
    for user_info in user_list_info
]

# Initialize variables
for user in user_list:
    user.agent.init_target_network()
    # (
    #     path="C:/Users/USER/PycharmProjects/mec_drl-masterr/mec_drl-master/mec_drl-master/data_set_OGD.npz"
    # )

@pytest.fixture
def env():
    # Create and return the environment object
    # Make sure to adjust this to properly create your environment instance
    return MecSvrEnv(user_list, NUM_R, SIGMA2, MAX_EPISODE_LEN)