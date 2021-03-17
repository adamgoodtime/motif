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
                score, fitness = test_with_hidden_layer(agent)
                if score >= 500:
                    print("complete with hidden")
                    # test_without_hidden_layer(agent, render=True)
            else:
                score, fitness = test_without_hidden_layer(agent)
                if score >= 500:
                    print("complete without hidden")
                    # test_without_hidden_layer(agent, render=True)
            scores.append(score)
            fitnesses.append(fitness)
        agent_scores.append(np.mean(scores))
        agent_fitness.append(np.mean(fitnesses))
        print("\nfinished agent", len(agent_scores), "\nfinal score", agent_scores[-1])
        print(scores)
        print("final fitness = ", agent_fitness[-1], "\n\n")
        print(fitnesses)

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

def test_with_hidden_layer(agent, render=False):

    in2rec, rec2rec, rec2out, in2out, in2rec_d, rec2rec_d, hidden_size, neuron_params = agent

    tau = neuron_params['tau']
    v_thresh = neuron_params['v_thresh']
    i_offset = neuron_params['i_offset']

    environment = gym.make("CartPole-v1")
    # env.reset()
    observation = environment.reset()
    # print(observation)
    cumulative_reward = 0.  # time_step.reward
    if neuron_type == 'tf_basic':
        neuron = BasicLIF(n_in=inputs,
                          n_rec=hidden_size,
                          weights_in=in2rec,
                          weights_rec=rec2rec,
                          tau=20., thr=0.615, dt=1., dtype=tf.float32,
                          dampening_factor=0.3,
                          non_spiking=False)
    elif neuron_type == 'tf_LIF':
        max_local_delay = int(max(np.max(in2rec_d), np.max(rec2rec_d)))
        neuron = LIF(n_in=inputs,
                     n_rec=hidden_size,
                     weights_in=in2rec,
                     weights_rec=rec2rec,
                     delays_in=in2rec_d,
                     delays_rec=rec2rec_d,
                     n_delay=max_local_delay,
                     tau=tau, thr=v_thresh, i_offset=i_offset,
                     dt=1., dtype=tf.float32,
                     dampening_factor=0.3)
    state = neuron.zero_state(1, tf.float32)

    all_z = []
    all_v = []
    all_env_state = []
    for t in range(exposure_time * 4):
        if render:
            environment.render()
        if penalise_oscillations:
            all_env_state.append(observation)
        observation = convert_observation_to_spikes(observation)
        if not velocity_info:
            observation[1] = 0
            observation[3] = 0
        observation = tf.expand_dims(tf.Variable(observation, dtype=tf.float32), axis=0)
        new_z, new_v, new_state = neuron.__call__(observation, state)
        # update/save state
        state = new_state
        all_z.append(new_z.numpy()[0])
        all_v.append(new_v.numpy()[0])
        rec_action = np.matmul(all_z[-1], rec2out)
        no_rec_action = np.matmul(observation, in2out)
        action = rec_action + no_rec_action
        # spikes_0 = all_z[-1][-2]
        # spikes_1 = all_z[-1][-1]
        # time_step = environment.step([spikes_0, spikes_1])
        if action[0][0] >= action[0][1]:
            binary_action = 1
        else:
            binary_action = 0
        observation, reward, done, info = environment.step(action=binary_action)
        cumulative_reward += reward
        # print(time_step, cumulative_reward)
        # print("outputs received", environment.send_state(), "- spikes:", spike_inputs, "- reward:", cumulative_reward)
        # print("reward:", cumulative_reward)
        if done and t < 499:
            print("final", observation)
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

def test_without_hidden_layer(agent, render=False):
    in2rec, rec2rec, rec2out, in2out, in2rec_d, rec2rec_d, hidden_size, neuron_params = agent

    environment = gym.make("CartPole-v1")
    # env.reset()
    observation = environment.reset()
    # print(observation)
    cumulative_reward = 0.  # time_step.reward

    all_z = []
    all_v = []
    all_env_state = []
    for t in range(exposure_time * 4):
        if render:
            environment.render()
        if penalise_oscillations:
            all_env_state.append(observation)
        observation = convert_observation_to_spikes(observation)
        observation = tf.expand_dims(tf.Variable(observation, dtype=tf.float32), axis=0)
        action = np.matmul(observation, in2out)
        # spikes_0 = all_z[-1][-2]
        # spikes_1 = all_z[-1][-1]
        # time_step = environment.step([spikes_0, spikes_1])
        if action[0][0] >= action[0][1]:
            binary_action = 1
        else:
            binary_action = 0
        observation, reward, done, info = environment.step(action=binary_action)
        cumulative_reward += reward
        # print(time_step, cumulative_reward)
        # print("outputs received", environment.send_state(), "- spikes:", spike_inputs, "- reward:", cumulative_reward)
        # print("reward:", cumulative_reward)
        if done and t < 499:
            print("final", observation)
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