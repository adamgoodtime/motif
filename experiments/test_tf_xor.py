from experiments.tf_xor import xor_env
from methods.basic_tf_neuron import BasicLIF
import tensorflow as tf
import matplotlib.pyplot as plt
from experiments.test_config import *

def test_xor(connections):
    agent_scores = []
    agent_test_result = []
    for agent in connections:
        in2rec, rec2rec, rec2out, in2out, hidden_size = agent
        environment = xor_env(spike_rates_off_on=[rate_off, rate_on],
                              stochastic=stochastic,
                              exposure_time=exposure_time,
                              negative=negative_reward)
        time_step = environment.reset()
        print(time_step)
        cumulative_reward = time_step.reward
        neuron = BasicLIF(n_in=inputs,
                          n_rec=hidden_size,
                          weights_in=in2rec,
                          weights_rec=rec2rec,
                          tau=20., thr=0.615, dt=1., dtype=tf.float32,
                          dampening_factor=0.3,
                          non_spiking=False)
        state = neuron.zero_state(1, tf.float32)

        all_z = []
        all_v = []
        for _ in range(exposure_time*4):
            # query env to get SNN input
            spike_inputs = environment.generate_spikes_out()
            # query SNN to get spikes out
            [new_z, new_v], new_state = neuron.__call__(spike_inputs, state)
            # update/save state
            state = new_state
            all_z.append(new_z.numpy()[0])
            all_v.append(new_v.numpy()[0])
            rec_action = np.matmul(all_z[-1], rec2out)
            no_rec_action = np.matmul(spike_inputs, in2out)
            action = rec_action + no_rec_action
            # spikes_0 = all_z[-1][-2]
            # spikes_1 = all_z[-1][-1]
            # time_step = environment.step([spikes_0, spikes_1])
            time_step = environment.step(action=action)
            cumulative_reward += time_step.reward
            # print(time_step, cumulative_reward)
            # print("outputs received", environment.send_state(), "- spikes:", spike_inputs, "- reward:", cumulative_reward)
            # print("reward:", cumulative_reward)
            if not time_step.discount:
                agent_test_result.append(sum(environment.test_correct))
                agent_scores.append(0)#cumulative_reward.tolist())
                print("\nfinished agent", len(agent_scores), "\nfinal score", agent_scores[-1])
                print("correct tests = ", environment.test_correct, agent_test_result[-1], "\n\n")
                break
    print("test conns")
    return agent_test_result#[agent_scores, agent_test_result]



