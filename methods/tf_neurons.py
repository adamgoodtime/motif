# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf

Cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell
# tfe = tf.contrib.eager
# rd = np.random.RandomState(3000)

def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    if not n_delay:
        return tf.expand_dims(w, axis=2)

    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.equal(d, kd)
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD

def tf_roll(buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
        shp = buffer.get_shape()
        l_shp = len(shp)

        if shp[-1] == 0:
            return buffer

        # Permute the index to roll over the right index
        perm = np.concatenate([[axis],np.arange(axis),np.arange(start=axis+1,stop=l_shp)])
        buffer = tf.transpose(buffer, perm=perm)

        # Add an element at the end of the buffer if requested, otherwise, add zero
        if new_last_element is None:
            shp = tf.shape(buffer)
            new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
        new_last_element = tf.expand_dims(new_last_element, axis=0)
        new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

        # Revert the index permutation
        inv_perm = np.argsort(perm)
        new_buffer = tf.transpose(new_buffer,perm=inv_perm)

        new_buffer = tf.identity(new_buffer,name='Roll')
        #new_buffer.set_shape(shp)
    return new_buffer

def einsum_bi_ijk_to_bjk(a,b):
    batch_size = tf.shape(a)[0]
    shp_a = a.get_shape()
    shp_b = b.get_shape()

    b_ = tf.reshape(b, (int(shp_b[0]), int(shp_b[1]) * int(shp_b[2])))
    ab_ = tf.matmul(a, b_)
    ab = tf.reshape(ab_, (batch_size, int(shp_b[1]), int(shp_b[2])))
    return ab

LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r'))


class CustomALIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=.615, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16, tag='',
                 stop_gradients=False, w_in_init=None, w_rec_init=None, n_refractory=1, rec=True):
        """
        CustomALIF provides the recurrent tensorflow cell model for implementing LSNNs in combination with
        eligibility propagation (e-prop).
        Cell output is a tuple: z (spikes): n_batch x n_neurons,
                                s (states): n_batch x n_neurons x 2,
                                diag_j (neuron specific jacobian, partial z / partial s): (n_batch x n_neurons x 2 x 2,
                                                                                           n_batch x n_neurons x 2),
                                partials_wrt_biases (partial s / partial input_current for inputs, recurrent spikes):
                                        (n_batch x n_neurons x 2, n_batch x n_neurons x 2)
        UPDATE: This model uses v^{t+1} ~ alpha * v^t + i_t instead of ... + (1 - alpha) * i_t
                it is therefore required to rescale thr, and beta of older version by
                thr = thr_old / (1 - exp(- 1 / tau))
                beta = beta_old * (1 - exp(- 1 / tau_adaptation)) / (1 - exp(- 1 / tau))
        UPDATE: refractory periods are implemented
        :param n_in: number of input neurons
        :param n_rec: number of output neurons
        :param tau: membrane time constant
        :param thr: spike threshold
        :param dt: length of discrete time steps
        :param dtype: data type of tensors
        :param dampening_factor: used in pseudo-derivative
        :param tau_adaptation: time constant of adaptive threshold decay
        :param beta: impact of adapting thresholds
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in_init: initial weights for input connections
        :param w_rec_init: initial weights for recurrent connections
        :param n_refractory: number of refractory time steps
        """

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
        with tf.variable_scope('CustomALIF_' + str(tag)):
            self.n_refractory = n_refractory
            self.tau_adaptation = tau_adaptation
            self.beta = beta
            self.decay_b = np.exp(-dt / tau_adaptation)

            if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
            if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

            tau = tf.cast(tau, dtype=dtype)
            dt = tf.cast(dt, dtype=dtype)
            self.rec = rec

            self.dampening_factor = dampening_factor
            self.stop_gradients = stop_gradients
            self.dt = dt
            self.n_in = n_in
            self.n_rec = n_rec
            self.data_type = dtype

            self._num_units = self.n_rec

            self.tau = tau
            self._decay = tf.exp(-dt / tau)
            self.thr = thr

            with tf.variable_scope('InputWeights'):
                # Input weights
                init_w_in_var = w_in_init if w_in_init is not None else \
                    (rd.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
                self.w_in_var = tf.get_variable("InputWeight", initializer=init_w_in_var, dtype=dtype)
                self.w_in_val = self.w_in_var

            with tf.variable_scope('RecWeights'):
                if rec:
                    init_w_rec_var = w_rec_init if w_rec_init is not None else \
                        (rd.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
                    self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
                    self.w_rec_val = self.w_rec_var

                    self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

                    # Disconnect autotapse
                    self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),self.w_rec_val)

                    dw_val_dw_var_rec = np.ones((self._num_units,self._num_units)) - np.diag(np.ones(self._num_units))
            dw_val_dw_var_in = np.ones((n_in,self._num_units))

            self.dw_val_dw_var = [dw_val_dw_var_in, dw_val_dw_var_rec] if rec else [dw_val_dw_var_in,]

            self.variable_list = [self.w_in_var, self.w_rec_var] if rec else [self.w_in_var,]
            self.built = True


    @property
    def state_size(self):
        return CustomALIFStateTuple(s=tf.TensorShape((self.n_rec, 2)), z=self.n_rec, r=self.n_rec)

    def set_weights(self, w_in, w_rec):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        if w_rec_rank == 3:
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(w_rec), w_rec)
        self.w_in_val = w_in

    @property
    def output_size(self):
        return [self.n_rec, tf.TensorShape((self.n_rec, 2)),
                [tf.TensorShape((self.n_rec, 2, 2)), tf.TensorShape((self.n_rec, 2))],
                [tf.TensorShape((self.n_rec, 2))] * 2]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        return CustomALIFStateTuple(s=s0, z=z0, r=r0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        decay = self._decay

        z = state.z
        s = state.s
        v, b = s[..., 0], s[..., 1]

        old_z = self.compute_z(v, b)

        if self.stop_gradients:
            z = tf.stop_gradient(z)

        new_b = self.decay_b * b + old_z

        if len(self.w_in_val.get_shape().as_list()) == 3:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            i_in = tf.matmul(inputs, self.w_in_val)
        if self.rec:
            if len(self.w_rec_val.get_shape().as_list()) == 3:
                i_rec = tf.einsum('bi,bij->bj', z, self.w_rec_val)
            else:
                i_rec = tf.matmul(z, self.w_rec_val)
            i_t = i_in + i_rec
        else:
            i_t = i_in

        I_reset = z * self.thr * self.dt

        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                 0., float(self.n_refractory))
        new_s = tf.stack((new_v, new_b), axis=-1)

        def safe_grad(y, x):
            g = tf.gradients(y, x)[0]
            if g is None:
                g = tf.zeros_like(x)
            return g

        dnew_v_ds = tf.gradients(new_v, s, name='dnew_v_ds')[0]
        dnew_b_ds = tf.gradients(new_b, s, name='dnew_b_ds')[0]
        dnew_s_ds = tf.stack((dnew_v_ds, dnew_b_ds), 2, name='dnew_s_ds')

        dnew_z_dnew_v = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_v))
        dnew_z_dnew_b = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_b))
        dnew_z_dnew_s = tf.stack((dnew_z_dnew_v, dnew_z_dnew_b), axis=-1)

        diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]

        # "in_weights, rec_weights"
        # ds_dW_bias: 2 x n_rec
        dnew_v_di = safe_grad(new_v,i_t)
        dnew_b_di = safe_grad(new_b,i_t)
        dnew_s_di = tf.stack([dnew_v_di,dnew_b_di], axis=-1)

        partials_wrt_biases = [dnew_s_di, dnew_s_di]

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r)
        return [new_z, new_s, diagonal_jacobian, partials_wrt_biases], new_state



class LIF(Cell):
    def __init__(self, n_in, n_rec, weights_in=[], weights_rec=[], delays_in=[], delays_rec=[], tau=20., thr=0.03,
                 dt=1., n_refractory=0, i_offset=None, dtype=tf.float32, n_delay=1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 dampening_factor=0.3,
                 injected_noise_current=0.,
                 V0=1.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.

        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
        """

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)
        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor

        # Parameters
        self.n_delay = n_delay
        self.n_refractory = n_refractory

        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tf.Variable(tau, dtype=dtype, name="Tau", trainable=False)
        self._decay = tf.exp(-dt / tau)
        self.thr = tf.Variable(thr, dtype=dtype, name="Threshold", trainable=False)
        if i_offset is None:
            self.i_offset = tf.zeros((n_rec, 1), dtype=dtype, name='i_offset')
        else:
            self.i_offset = tf.Variable(i_offset, dtype=dtype, name='i_offset')

        self.V0 = V0
        self.injected_noise_current = injected_noise_current

        # self.rewiring_connectivity = rewiring_connectivity
        self.in_neuron_sign = in_neuron_sign
        self.rec_neuron_sign = rec_neuron_sign

    # with tf.variable_scope('InputWeights'):
        self.w_in_var = tf.Variable(weights_in, dtype=dtype, name="InputWeight")
        self.w_in_val = self.w_in_var

        self.w_in_val = self.V0 * self.w_in_val
        self.w_in_delay = tf.Variable(delays_in, dtype=tf.int64, name="InDelays", trainable=False)
        self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

    # with tf.variable_scope('RecWeights'):
        self.w_rec_var = tf.Variable(weights_rec, dtype=dtype, name='RecurrentWeight')
        self.w_rec_val = self.w_rec_var

        # recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

        self.w_rec_val = self.w_rec_val * self.V0
        # self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
        #                           self.w_rec_val)  # Disconnect autotapse
        self.w_rec_delay = tf.Variable(delays_rec, dtype=tf.int64, name="RecDelays", trainable=False)
        self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    @property
    def state_size(self):
        return LIFStateTuple(v=self.n_rec,
                             z=self.n_rec,
                             i_future_buffer=(self.n_rec, self.n_delay),
                             z_buffer=(self.n_rec, self.n_refractory))

    @property
    def output_size(self):
        return self.n_rec

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, max(1, self.n_delay)), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return LIFStateTuple(
            v=v0,
            z=z0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        i_future_buffer = state.i_future_buffer
        i_in = einsum_bi_ijk_to_bjk(inputs, self.W_in)
        i_rec = einsum_bi_ijk_to_bjk(state.z, self.W_rec)
        i_future_buffer += i_in + i_rec

        new_v, new_z = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = LIFStateTuple(v=new_v,
                                  z=new_z,
                                  i_future_buffer=new_i_future_buffer,
                                  z_buffer=new_z_buffer)
        return new_z, new_v, new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, decay=None, n_refractory=None, add_current=0.):
        """
        Function that generate the next spike and voltage tensor for given cell state.
        :param v
        :param z
        :param z_buffer:
        :param i_future_buffer:
        :param thr:
        :param decay:
        :param n_refractory:
        :param add_current:
        :return:
        """

        if self.injected_noise_current > 0:
            add_current = tf.random_normal(shape=z.shape, stddev=self.injected_noise_current)
        else:
            add_current = tf.zeros_like(z)
        add_current += self.i_offset

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            I_reset = z * thr * self.dt

            new_v = decay * v + (1 - decay) * i_t - I_reset

            # Spike generation
            v_scaled = (v - thr) / thr

            # new_z = differentiable_spikes(v_scaled=v_scaled)
            new_z = SpikeFunction(v_scaled, self.dampening_factor)

            if n_refractory > 0:
                is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), 0)
                new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)

            new_z = new_z * 1 / self.dt

            return new_v, new_z


LightLIFStateTuple = namedtuple('LightLIFStateTuple', ('v', 'z'))
class LightLIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.615, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 stop_z_gradients=False):
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
        self.stop_z_gradients = stop_z_gradients

        self._num_units = self.n_rec

        self.tau = tf.constant(tau, dtype=dtype)
        self._decay = tf.exp(-dt / self.tau)
        self.thr = thr

        with tf.variable_scope('InputWeights'):
            self.w_in_var = tf.Variable(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype)
            self.w_in_val = tf.identity(self.w_in_var)

        with tf.variable_scope('RecWeights'):
            self.w_rec_var = tf.Variable(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype)
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var),
                                      self.w_rec_var)  # Disconnect autotapse

    @property
    def state_size(self):
        return LightLIFStateTuple(v=self.n_rec, z=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return LightLIFStateTuple(v=v0, z=z0)

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        thr = self.thr
        z = state.z
        v = state.v
        decay = self._decay

        if self.stop_z_gradients:
            z = tf.stop_gradient(z)

        # update the voltage
        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z, self.w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + (1 - decay) * i_t - I_reset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt
        new_state = LightLIFStateTuple(v=new_v, z=new_z)
        return [new_z, new_v], new_state


LightALIFStateTuple = namedtuple('LightALIFState', (
    'z',
    'v',
    'b'))


class LightALIF(LightLIF):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6,
                 stop_z_gradients=False):

        super(LightALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt,
                                        dtype=dtype, dampening_factor=dampening_factor,
                                        stop_z_gradients=stop_z_gradients)
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

    @property
    def state_size(self):
        return LightALIFStateTuple(v=self.n_rec, z=self.n_rec, b=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        b0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        return LightALIFStateTuple(v=v0, z=z0, b=b0)


    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        z = state.z
        v = state.v
        decay = self._decay

        # the eligibility traces of b see the spike of the own neuron
        new_b = self.decay_b * state.b + (1. - self.decay_b) * z
        thr = self.thr + new_b * self.beta
        if self.stop_z_gradients:
            z = tf.stop_gradient(z)

        # update the voltage
        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z, self.w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + (1 - decay) * i_t - I_reset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt

        new_state = LightALIFStateTuple(v=new_v,z=new_z, b=new_b)
        return [new_z, new_v, new_b], new_state