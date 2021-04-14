from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from experiments.test_config import *

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class logic_env(py_environment.PyEnvironment):

    def __init__(self, exposure_time=10, spike_rates_off_on=[0, 10], stochastic=True, negative=True,
                 truth_table=[0, 1, 1, 0]):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=100, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=len(truth_table)-1, name='observation')
        self._state = 0
        self._correct_output = truth_table
        self._no_inputs = int(np.ceil(np.log2(len(truth_table))))
        self._possible_states = []
        for i in range(len(truth_table)):
            this_input = []
            value = i
            for j in range(self._no_inputs):
                this_input.append(int(value / np.power(2, self._no_inputs - j - 1)))
                value -= np.power(2, self._no_inputs - j - 1) * this_input[-1]
            self._possible_states.append(this_input)
        self._episode_ended = False
        self._dt = 1.
        self._exposure_time = exposure_time
        self._current_time = 0
        self._spike_rates_off_on = spike_rates_off_on
        self.stochastic = stochastic
        self.actions_in_state = [0, 0]
        self.test_correct = []
        self.negative = negative

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_time = 0
        self._state = 0
        self._episode_ended = False
        self.actions_in_state = [0, 0]
        return ts.restart(np.array([self._state], dtype=np.int32))

    def send_state(self):
        if self._state >= len(self._possible_states):
            self._reset()
        return self._possible_states[self._state]

    def generate_spikes_out(self, dtype=tf.float32):
        output_spikes = [[0], [0]]
        for id, output in enumerate(self._possible_states[self._state]):
            if self.stochastic:
                if np.random.random() < (self._dt / 1000.) * self._spike_rates_off_on[output]:
                    output_spikes[id] = 1
                else:
                    output_spikes[id] = 0
            else:
                if self._current_time % int(1000 / self._spike_rates_off_on[output]) == 0:
                    output_spikes[id] = 1
                else:
                    output_spikes[id] = 0
        output_spikes_tf = tf.Variable(output_spikes, dtype=dtype)
        return output_spikes_tf

    def get_final_performance(self):
        return self.test_correct

    def _step(self, action):
        action_0, action_1 = action
        self.actions_in_state[0] += action_0
        self.actions_in_state[1] += action_1
        # print("state {} - time {} - action {} - out* {}".format(self._state, self._current_time, action,
        #                                                         self._correct_output[self._state]))
        # Make sure episodes don't go on forever.
        if (self.actions_in_state[0] > self.actions_in_state[1] and 0 == self._correct_output[self._state]) or \
                (self.actions_in_state[1] > self.actions_in_state[0] and 1 == self._correct_output[self._state]):
            reward = 1
        else:
            reward = 0
            # raise ValueError('`action` should be 0 or 1.')
        if ((self.actions_in_state[0] < self.actions_in_state[1] and 0 == self._correct_output[self._state]) or
            (self.actions_in_state[1] < self.actions_in_state[0] and 1 == self._correct_output[self._state])) \
                and self.negative:
            reward = -1

        self._current_time += self._dt
        if self._current_time >= self._exposure_time:
            self._current_time = 0
            self.test_correct.append(reward)
            self.actions_in_state = [0, 0]
            self._state += 1

        if self._state >= len(self._correct_output):
            if self._episode_ended:
                return self.reset()
            self._episode_ended = True
            # return self._reset()

        # if self._episode_ended:
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(
        #         np.array([self._state], dtype=np.int32), reward=reward, discount=1.0)
        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward=reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=reward, discount=1.0)
        # return 0, self._state, reward, self._dt

if __name__ == "__main__":
    environment = logic_env()
    utils.validate_py_environment(environment, episodes=5)

    out_zero = np.array(0, dtype=np.int32)
    out_one = np.array(1, dtype=np.int32)

    environment = logic_env()
    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    for _ in range(50):
        # if np.random.random() > 0.5:
        #     time_step = environment.step(out_zero)
        # else:
        #     time_step = environment.step(out_one)
        spikes_0 = np.random.randint(100)
        spikes_1 = np.random.randint(100)
        time_step = environment.step([spikes_0, spikes_1])
        cumulative_reward += time_step.reward
        print(time_step, cumulative_reward)
        print("outputs received", environment.send_state())
        if not time_step.discount:
            break

    # time_step = environment.step(end_round_action)
    # print(time_step)
    cumulative_reward += time_step.reward
    print('Final Reward = ', cumulative_reward)
