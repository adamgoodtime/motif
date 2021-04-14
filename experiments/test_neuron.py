from methods.tf_neurons import *
import matplotlib.pyplot as plt

inputs = 2
hidden_size = 2

weight_test = 15
in2rec = [[0 for i in range(inputs)] for j in range(hidden_size)]
in2rec[0][0] = 12
in2rec[0][1] = 12
rec2rec = np.zeros((hidden_size, hidden_size))
max_delay = 1
in2rec_d = np.random.randint(0, max_delay, (inputs, hidden_size))  # np.ones((inputs, hidden_size))
rec2rec_d = np.random.randint(0, max_delay, (hidden_size, hidden_size))  # np.ones((hidden_size, hidden_size))

i_offset = [0 for i in range(hidden_size)]

neuron = LIF(n_in=inputs,
             n_rec=hidden_size,
             weights_rec=rec2rec,
             delays_rec=rec2rec_d,
             n_delay=max_delay,
             i_offset=i_offset,
             tau=20., thr=[0.615, 0.3], dt=1., dtype=tf.float32,
             dampening_factor=0.3)

state = neuron.zero_state(1, tf.float32)
all_i = []
currents = range(200)
runtime = 10000
input_current = []
rate = []
for c in currents:
    current = c / 10
    all_z = []
    all_v = []
    for i in range(runtime):
        # if i % 100 == 0:
        #     input_to_n = tf.expand_dims(tf.Variable([1, 1], dtype=tf.float32), axis=0)
        # else:
        input_to_n = tf.expand_dims(tf.Variable([current, current], dtype=tf.float32), axis=0)
        neuron.i_offset = input_to_n
        new_z, new_v, new_state = neuron.__call__(state)
        state = new_state
        all_z.append(new_z.numpy()[0][0])
        all_v.append(new_v.numpy()[0])
    input_current.append(current)
    rate.append([current, np.count_nonzero(all_z) / (runtime/1000.)])
    print("for current", current, "spike rate was", np.count_nonzero(all_z) / (runtime/1000.))


plt.figure()
# all_v = np.transpose(all_v)
# plt.plot([v[0] for v in all_v])
# plt.plot(range(len(all_v[0])), all_v[0])
# plt.plot(range(len(all_v[1])), all_v[1])
# plt.plot(all_v)
plt.plot(input_current, rate)
# plt.plot(all_v[1])
# a = range(len(all_z))
# plt.scatter([a,a,a,a], all_z)
plt.show()

print("done")