from experiments.tf_logic import logic_env
from methods.basic_tf_neuron import BasicLIF
from methods.tf_neurons import LIF
import tensorflow as tf
import matplotlib.pyplot as plt
from experiments.test_config import *

def test_logic(connections):
    agent_scores = []
    agent_test_result = []
    for agent in connections:
        score, result = agent_logic_test(agent)

        agent_test_result.append(sum(result))
        if agent_test_result[-1] == len(truth_table):
            print("completed test")
        agent_scores.append(score)  # cumulative_reward.tolist())
        print("\nfinished agent", len(agent_scores), "\nfinal score", agent_scores[-1])
        print("correct tests = ", result, agent_test_result[-1], "\n\n")
    print("test conns")
    return agent_test_result#[agent_scores, agent_test_result]

def agent_logic_test(agent):
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

    environment = logic_env(spike_rates_off_on=[rate_off, rate_on],
                            stochastic=stochastic,
                            exposure_time=exposure_time,
                            truth_table=truth_table,
                            negative=negative_reward)
    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward
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
    for _ in range(exposure_time * len(truth_table)):
        # query env to get SNN input
        # spike_inputs = environment.generate_spikes_out()
        # query SNN to get spikes out
        observation = generate_currents(environment)
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

        time_step = environment.step(action=action)
        cumulative_reward += time_step.reward
        # print(time_step, cumulative_reward)
        # print("outputs received", environment.send_state(), "- spikes:", spike_inputs, "- reward:", cumulative_reward)
        # print("reward:", cumulative_reward)
        if not time_step.discount:
            break
    return cumulative_reward, environment.test_correct

def generate_currents(env):
    current_env_state = env._possible_states[env._state]
    current = []
    for curr in current_env_state:
        if curr == 0:
            current.append(0.7)
        else:
            current.append(max_current)
    return current
