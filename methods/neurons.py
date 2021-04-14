import numpy as np
from numpy.core import multiarray
import math
import itertools
from copy import deepcopy
# import operator
# from spinn_front_end_common.utilities.globals_variables import get_simulator
# import traceback
# import csv
# from ast import literal_eval

class neuron_population(object):
    def __init__(self,
                 neuron_type='IF_cond_exp',
                 default=False,
                 io_prob=0.5,
                 inputs=0,
                 outputs=0,
                 non_spiking_out=True,
                 ex_prob=0.5,
                 read_population=False,
                 keep_reading=0,
                 pop_size=200
                 ):

        if neuron_type == 'tf_basic':
            v_thresh = 0.615
            v_thresh_stdev = v_thresh / 5.
            tau = 20
            tau_stdev = tau / 5.
            i_offset = 0
            i_offset_stdev = 0
        elif neuron_type == 'tf_LIF':
            v_thresh = 0.615
            v_thresh_stdev = v_thresh / 5.
            tau = 20.
            tau_stdev = tau / 5.
            i_offset = 0.
            i_offset_stdev = 0.2
        else:
            print("incorrect neuron type selected")
            raise Exception

        # neuron params
        self.v_rest = 0
        self.v_rest_stdev = 0
        self.tau = tau
        self.tau_stdev = tau_stdev
        self.v_thresh = v_thresh
        self.v_thresh_stdev = v_thresh_stdev
        self.v_reset = 0
        self.v_reset_stdev = 0
        self.i_offset = i_offset
        self.i_offset_stdev = i_offset_stdev
        self.v = 0
        self.v_stdev = 0

        self.io_prob = io_prob
        self.outputs = outputs
        self.inputs = inputs
        self.ex_prob = ex_prob
        self.pop_size = pop_size
        self.default = default
        self.non_spiking_out = non_spiking_out

        self.neuron_params = {}
        self.neuron_params['v_rest'] = self.v_rest
        self.neuron_params['tau'] = self.tau
        self.neuron_params['v_thresh'] = self.v_thresh
        self.neuron_params['v_reset'] = self.v_reset
        self.neuron_params['i_offset'] = self.i_offset
        self.neuron_params['v'] = self.v

        self.neuron_param_stdevs = {}
        self.neuron_param_stdevs['v_rest'] = self.v_rest_stdev
        self.neuron_param_stdevs['tau'] = self.tau_stdev
        self.neuron_param_stdevs['v_thresh'] = self.v_thresh_stdev
        self.neuron_param_stdevs['v_reset'] = self.v_reset_stdev
        self.neuron_param_stdevs['i_offset'] = self.i_offset_stdev
        self.neuron_param_stdevs['v'] = self.v_stdev

        if self.default:
            # self.neuron_param_stdevs = {}
            # self.neuron_params = {}
            self.neuron_params['i_offset'] = 0
            for param in self.neuron_param_stdevs:
                self.neuron_param_stdevs[param] = 0
            self.pop_size = self.inputs + self.outputs + 1

        self.neuron_configs = {}
        self.neurons_generated = -1  # counting backwards to avoid any confusion with motifs
        self.total_weight = 0

        if read_population:
            self.read_population = read_population
            self.load_neurons()
            print("reading the population")
        else:
            if not self.default:
                neurons_to_create = int(self.pop_size * self.io_prob)
            else:
                neurons_to_create = self.inputs + self.outputs
            io_choice = -1
            for i in range(neurons_to_create):
                neuron = {}
                neuron['id'] = '{}'.format(self.neurons_generated)
                if not self.default:
                    io_choice = np.random.randint(0, self.inputs + self.outputs)
                else:
                    io_choice += 1
                if io_choice - self.inputs < 0:
                    neuron['type'] = 'input'
                    neuron['io'] = io_choice
                else:
                    neuron['type'] = 'output'
                    neuron['io'] = io_choice - self.inputs
                if self.default:
                    neuron['weight'] = (1. / float(self.inputs + self.outputs)) * self.io_prob
                else:
                    neuron['weight'] = float(self.pop_size) / float(self.inputs + self.outputs)
                    neuron['weight'] *= self.io_prob
                neuron['params'] = {}
                if not self.default:
                    for param in self.neuron_params:
                        if neuron['type'] == 'output' and self.non_spiking_out and param == 'v_thresh':
                            neuron['params'][param] = self.non_spiking_out
                        else:
                            neuron['params'][param] = np.random.normal(self.neuron_params[param],
                                                                       self.neuron_param_stdevs[param])
                self.insert_neuron(neuron, check=False, weight=neuron['weight'])
            if not self.default:
                neurons_to_create = int(self.pop_size * (1. - self.io_prob))
            else:
                neurons_to_create = self.pop_size - (self.inputs + self.outputs)
            for i in range(neurons_to_create):
                not_new = True
                while not_new:
                    neuron = {}
                    neuron['id'] = '{}'.format(self.neurons_generated)
                    neuron['io'] = False
                    neuron['type'] = 'hidden'
                    neuron['params'] = {}
                    if not self.default:
                        for param in self.neuron_params:
                            neuron['params'][param] = np.random.normal(self.neuron_params[param], self.neuron_param_stdevs[param])
                    if self.inputs + self.outputs > 0:
                        base_weight = float(self.pop_size) / float(self.pop_size - self.inputs - self.outputs)
                        base_weight *= (1. - self.io_prob)
                    else:
                        base_weight = 1
                    if self.default:
                        neuron['weight'] = 1. - self.io_prob
                    else:
                        neuron['weight'] = base_weight

                    not_new = self.check_neuron(neuron)
                self.insert_neuron(neuron, check=False, weight=neuron['weight'])

    '''Keeps looping through possible neuron configurations until a neuron not currently in the population is created'''
    def generate_neuron(self, weight=0):
        if self.default:
            return self.choose_neuron()
        found_new = False
        while not found_new:
            neuron = {}
            if np.random.random() < self.io_prob:
                io_choice = np.random.randint(self.inputs + self.outputs)
                if io_choice - self.inputs < 0:
                    neuron['type'] = 'input'
                    neuron['io'] = io_choice
                else:
                    neuron['type'] = 'output'
                    neuron['io'] = io_choice - self.inputs
            else:
                neuron['io'] = False
                neuron['type'] = 'hidden'
            neuron['params'] = {}
            for param in self.neuron_params:
                if neuron['type'] == 'output' and self.non_spiking_out and param == 'v_thresh':
                    neuron['params'][param] = self.non_spiking_out
                else:
                    neuron['params'][param] = np.random.normal(self.neuron_params[param],
                                                               self.neuron_param_stdevs[param])
            neuron['weight'] = weight
            neuron['id'] = '{}'.format(self.neurons_generated)
            if not self.check_neuron(neuron):
                neuron_id = self.insert_neuron(neuron, check=False, weight=weight)
                found_new = True
        return neuron_id

    '''Checks if there are similar neurons before inserting and either returns the id of the similar one or the new id 
    of the inserted one'''
    def insert_neuron(self, neuron, weight=0, check=True, read=False):
        self.total_weight = 0
        if read:
            weight = neuron['weight']
            neuron_id = neuron['id']
        else:
            neuron_id = '{}'.format(self.neurons_generated)
            does_it_exist = True
            while does_it_exist:
                try:
                    does_it_exist = self.neuron_configs['{}'.format(self.neurons_generated)]
                    print(self.neurons_generated, "existed")
                    self.neurons_generated -= 1
                except:
                    # traceback.print_exc()
                    neuron_id = '{}'.format(self.neurons_generated)
                    does_it_exist = False
        if check:
            repeat_check = self.check_neuron(neuron)
            if repeat_check:
                return repeat_check
            else:
                # self.total_weight += neuron['weight']
                neuron['weight'] = weight
                self.neuron_configs[neuron_id] = neuron
                self.neurons_generated -= 1
        else:
            # self.total_weight += neuron['weight']
            neuron['weight'] = weight
            self.neuron_configs[neuron_id] = neuron
            self.neurons_generated -= 1
        return '{}'.format(self.neurons_generated + 1)

    '''Returns false if no neurons are the same and the id if there is one similar'''
    def check_neuron(self, new_neuron):
        for neuron in self.neuron_configs:
            the_same = True
            for param in self.neuron_configs[neuron]:
                if self.neuron_configs[neuron][param] == new_neuron[param] or param == 'id' or param == 'weight':
                    the_same = True
                else:
                    the_same = False
                    break
            if the_same:
                return self.neuron_configs[neuron]['id']
        return False

    def choose_neuron(self):
        if not self.total_weight:
            for neuron in self.neuron_configs:
                self.total_weight += self.neuron_configs[neuron]['weight']
        choice = np.random.random() * float(self.total_weight)
        for neuron in self.neuron_configs:
            choice -= self.neuron_configs[neuron]['weight']
            if choice < 0:
                return neuron

    def shift_io(self, neuron_id, in_or_out, direction):
        neuron = deepcopy(self.neuron_configs[neuron_id])
        if in_or_out == 'in':
            if neuron['type'] == 'input':
                neuron['io'] += direction
                neuron['io'] %= self.inputs
                return self.insert_neuron(neuron)
            else:
                return neuron_id
        else:
            if neuron['type'] == 'output':
                neuron['io'] += direction
                neuron['io'] %= self.outputs
                return self.insert_neuron(neuron)
            else:
                return neuron_id

    def save_neurons(self, iteration, config):
        np.save('Neuron pop {} {}.npy'.format(iteration, config), self.neuron_configs)

    def load_neurons(self):
        file_name = self.read_population
        read_neurons = np.load(file_name)
        read_neurons = read_neurons.tolist()
        for neuron_id in read_neurons:
            self.insert_neuron(read_neurons[neuron_id], read=True)

    def reset_weights(self):
        self.total_weight = 0
        for neuron in self.neuron_configs:
            self.neuron_configs[neuron]['weight'] = 0.0

    def update_weights(self, neuron_ids, weight):
        for neuron in neuron_ids:
            self.neuron_configs[neuron]['weight'] += weight

    def average_weights(self, neuron_counts):
        for neuron in neuron_counts:
            self.neuron_configs[neuron]['weight'] /= neuron_counts[neuron]

    def clean_population(self):
        if not self.default:
            to_be_deleted = []
            for neuron_id in self.neuron_configs:
                if self.neuron_configs[neuron_id]['weight'] == 0 \
                        and self.neuron_configs[neuron_id]['type'] != 'input'\
                        and self.neuron_configs[neuron_id]['type'] != 'output':
                    to_be_deleted.append(neuron_id)
            for id in to_be_deleted:
                self.delete_neuron(id)

    def delete_neuron(self, neuron_id):
        del self.neuron_configs[neuron_id]

    def return_neuron(self, neuron_id):
        return self.neuron_configs[neuron_id]




