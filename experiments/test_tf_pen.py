from experiments.tf_xor import xor_env
from methods.basic_tf_neuron import BasicLIF
from methods.tf_neurons import LIF
import tensorflow as tf
import matplotlib.pyplot as plt
from experiments.test_config import *
import gym

def test_pen(connections):
    agent_scores = []
    agent_fitness = []
    for agent in connections:
        scores = []
        fitnesses = []
        for i in range(repeat_test):
            if agent[-1]:
                score, fitness = agent_pendulum_test(agent)
                if score >= 500:
                    print("complete with hidden")
                    # test_default_neurons(agent, render=True)
            else:
                score, fitness = agent_pendulum_test(agent)
                if score >= 500:
                    print("complete without hidden")
                    # test_default_neurons(agent, render=True)
            scores.append(score)
            fitnesses.append(fitness)
        agent_scores.append(np.mean(scores))
        agent_fitness.append(np.mean(fitnesses))
        print("\nfinished agent", len(agent_scores)-1, "\nfinal score", agent_scores[-1])
        print(scores)
        print(fitnesses)
        print("final fitness = ", agent_fitness[-1], "\n\n\n")

    print("tested conns")
    return agent_scores#[agent_scores, agent_test_result]

def convert_observation_to_spikes(observation):
    pos = observation[0]
    p_v = observation[1]
    ang = observation[2]
    a_v = observation[3]
    max_value = 2.4
    min_value = -2.4
    field_width = (max_value - min_value) / (receptive_fields - 1)
    bins = []
    for i in range(receptive_fields):
        bins.append(min_value + (i * field_width))
    new_observation = []
    for value in observation:
        for field in bins:
            dist = min(value - field, receptive_width)
            firing_prob = 1 - (np.abs(dist) / receptive_width)
            did_it_fire = int(np.random.random() < (firing_prob * (max_pen_rate / 1000.)))
            new_observation.append(did_it_fire)
    return new_observation

def normalise_inputs(observation):
    pos = observation[0]
    p_v = observation[1]
    ang = observation[2]
    a_v = observation[3]
    norm_observation = []
    max_value = 2.4
    min_value = -2.4
    if not velocity_info:
        observation = [observation[0], observation[2]]
    if receptive_fields > 1:
        field_width = (max_value - min_value) / (receptive_fields - 1)
        bins = []
        for i in range(receptive_fields):
            bins.append(min_value + (i * field_width))
        for ob in observation:
            for bin in bins:
                dist = max(0, 1. - abs(bin - ob) / receptive_width)
                norm_observation.append(dist * max_current)
        return norm_observation
    ob_range = max_value - min_value
    for ob in observation:
        norm_observation.append(((ob - min_value) / ob_range) * max_current)
    return norm_observation

def agent_pendulum_test(agent, render=False):
    conn_matrix, delay_matrix, indexed_i, indexed_o, neuron_params = agent
    hidden_size = len(conn_matrix)

    if neuron_params:
        tau = neuron_params['tau']
        v_thresh = neuron_params['v_thresh']
        i_offset = neuron_params['i_offset']
    else:
        tau = 20.
        v_thresh = 0.615
        i_offset = 0.

    environment = gym.make("CartPole-v1")
    # env.reset()
    observation = environment.reset()
    # print(observation)
    cumulative_reward = 0.  # time_step.reward
    if neuron_type == 'tf_basic':
        neuron = BasicLIF(n_in=inputs,
                          n_rec=hidden_size,
                          weights_rec=conn_matrix,
                          tau=20., thr=0.615, dt=1., dtype=tf.float32,
                          dampening_factor=0.3,
                          non_spiking=False)
    elif neuron_type == 'tf_LIF':
        max_local_delay = int(np.max(delay_matrix))
        neuron = LIF(n_in=inputs,
                     n_rec=hidden_size,
                     weights_rec=conn_matrix,
                     delays_rec=delay_matrix,
                     n_delay=max_local_delay,
                     tau=tau, thr=v_thresh, i_offset=i_offset,
                     dt=1., dtype=tf.float32,
                     dampening_factor=0.3)
    state = neuron.zero_state(1, tf.float32)

    all_z = []
    all_v = []
    all_env_state = []
    # spike_count = 0
    for t in range(exposure_time * 4):
        if render:
            environment.render()
        if penalise_oscillations:
            all_env_state.append(observation)
        # observation = convert_observation_to_spikes(observation)
        observation = normalise_inputs(observation)
        # observation = tf.expand_dims(tf.Variable(observation, dtype=tf.float32), axis=0)
        current_input = []
        for i in range(hidden_size):
            if i in indexed_i:
                in_index = indexed_i[i]
                current_input.append(observation[in_index])
            else:
                current_input.append(0.)
        neuron.i_offset = current_input
        new_z, new_v, new_state = neuron.__call__(state)
        # update/save state
        state = new_state
        all_z.append(new_z.numpy()[0])
        all_v.append(new_v.numpy()[0])
        # spike_count += sum(all_z[-1])
        action = [0.00001, 0.00001]
        if spike_controlled:
            for idx, z in enumerate(all_z[-1]):
                if idx in indexed_o:
                    if z > 0:
                        print("z", end=' ')
                    action[indexed_o[idx]] += z
        else:
            for idx, v in enumerate(all_v[-1]):
                if idx in indexed_o:
                    if v > 0:
                        print("v", end=' ')
                    action[indexed_o[idx]] += v
        # spikes_0 = all_z[-1][-2]
        # spikes_1 = all_z[-1][-1]
        # time_step = environment.step([spikes_0, spikes_1])
        choice = action[0] / (action[1] + action[0])
        # if np.random.random() < choice:
        if choice < 0.5:
            binary_action = 1
        else:
            binary_action = 0
        observation, reward, done, info = environment.step(action=binary_action)
        cumulative_reward += reward
        # print(time_step, cumulative_reward)
        # print("outputs received", environment.send_state(), "- spikes:", spike_inputs, "- reward:", cumulative_reward)
        # print("reward:", cumulative_reward)
        if done and t < 499:
            print("final", cumulative_reward)
            print("total spikes -", np.sum(all_z, axis=0))
            print("average firing rate -", np.round(np.average(all_z, axis=0) / (t/1000), 1))
            # agent_test_result.append(cumulative_reward)
            # agent_scores.append(cumulative_reward)
            break

    penalty = 0.
    for i in range(min(t, pen_cutoff)):
        penalty += np.abs(all_env_state[-1 - i][0]) + np.abs(all_env_state[-1 - i][1]) + \
                   np.abs(all_env_state[-1 - i][2]) + np.abs(all_env_state[-1 - i][3])
    f2 = 0.75 / penalty
    f1 = cumulative_reward / 1000
    f = (f1 * 0.1) + (f2 * 0.9)


    return cumulative_reward, f