# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf

Cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell
# tfe = tf.contrib.eager

def SpikeFunction(v_scaled, dampening_factor, non_spiking=False):
    if not non_spiking:
        z_ = tf.greater(v_scaled, 0.)
    else:
        z_ = tf.greater(0., 1.)
    z_ = tf.cast(z_, dtype=tf.float32)

    # def grad(dy):
    #     dE_dz = dy
    #     dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
    #     dz_dv_scaled *= dampening_factor
    #
    #     dE_dv_scaled = dE_dz * dz_dv_scaled
    #
    #     return [dE_dv_scaled,
    #             tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction")#, grad

BasicLIFStateTuple = namedtuple('BasicLIFStateTuple', ('v', 'z'))
class BasicLIF(Cell):
    def __init__(self, n_in, n_rec, weights_in=[], weights_rec=[], tau=20., thr=0.615, i_offset=0., dt=1., dtype=tf.float32, dampening_factor=0.3,
                 non_spiking=False):
        '''
        A tensorflow RNN cell model to simulate Learky Integrate and Fire (LIF) neurons.

        WARNING: This model might not be compatible with tensorflow framework extensions because the input and recurrent
        weights are defined with tf.Variable at creation of the cell instead of using variable scopes.

        :param n_in: number of input neurons
        :param n_rec: number of recurrenet neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step
        :param dtype: data type
        :param dampening_factor: parameter to stabilize learning
        :param stop_z_gradients: if true, some gradients are stopped to get an equivalence between eprop and bptt
        '''

        self.dampening_factor = dampening_factor
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.non_spiking = non_spiking

        self._num_units = self.n_rec

        self.tau = tf.constant(tau, dtype=dtype)
        self._decay = tf.exp(-dt / self.tau)
        self.thr = thr
        self.i_offset = i_offset

        with tf.compat.v1.variable_scope('InputWeights'):
            # self.w_in_var = tf.Variable(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype)
            # self.w_in_val = tf.identity(self.w_in_var)
            self.w_in_var = tf.Variable(weights_in, dtype=dtype)
            self.w_in_val = tf.Variable(weights_in, dtype=dtype)

        with tf.compat.v1.variable_scope('RecWeights'):
            # self.w_rec_var = tf.Variable(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype)
            # self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            # self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var),
            #                           self.w_rec_var)  # Disconnect autotapse
            self.w_rec_var = tf.Variable(weights_rec, dtype=dtype)
            self.w_rec_val = tf.Variable(weights_rec, dtype=dtype)

    @property
    def state_size(self):
        return BasicLIFStateTuple(v=self.n_rec, z=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return BasicLIFStateTuple(v=v0, z=z0)

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        thr = self.thr
        z = state.z
        v = state.v
        decay = self._decay

        # if self.stop_z_gradients:
        #     z = tf.stop_gradient(z)

        # update the voltage
        # i_t = tf.matmul([inputs, inputs], [self.w_in_val, self.w_in_val])# + tf.matmul(z, self.w_rec_val)
        # i_t = tf.matmul(tf.expand_dims(inputs, axis=0), tf.expand_dims(self.w_in_val, axis=-1))# + tf.matmul(z, self.w_rec_val)
        i_t = tf.matmul(tf.expand_dims(tf.Variable(inputs, dtype=dtype), axis=0), self.w_in_val)
        # i_t = tf.matmul(tf.Variable(inputs, dtype=dtype), self.w_in_val)
        i_t_rec = tf.matmul(z, self.w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + (1 - decay) * (i_t + i_t_rec) - I_reset# + self.i_offset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor, non_spiking=self.non_spiking)
        new_z = new_z * 1 / self.dt
        new_state = BasicLIFStateTuple(v=new_v, z=new_z)
        return [new_z, new_v], new_state