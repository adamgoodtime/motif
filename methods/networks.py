import spynnaker8 as p
from spynnaker.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
from spinn_front_end_common.utilities.globals_variables import get_simulator

import pylab
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import \
    SpynnakerExternalDevicePluginManager as ex
import sys, os
import time
import socket
import numpy as np
import math


class motif_population(object):
    def __init__(self,
                 max_motif_size=4,
                 min_motif_size=2,
                 population_size=200,
                 static_population=True,
                 population_seed=None,
                 read_entire_population=False,
                 weights=True,
                 weight_range=(-0.1, 0.1),
                 initial_weight=0,
                 weight_stdev=0.02,
                 delays=True,
                 delay_range=(0, 25),
                 delay_stdev=3,
                 initial_hierarchy_depth=1,
                 io_config='fixed',  # fixed, dynamic/coded probabilistic, uniform
                 multi_synapse=True):

        self.max_motif_size = max_motif_size
        self.min_motif_size = min_motif_size
        self.population_size = population_size
        self.static_population = static_population
        self.population_seed = population_seed
        # self.read_entire_population = read_entire_population
        self.weights = weights
        self.weight_range = weight_range
        self.initial_weight = initial_weight
        self.weight_stdev = weight_stdev
        self.delays = delays
        self.delay_range = delay_range
        self.delay_stdev = delay_stdev
        self.initial_hierarchy_depth = initial_hierarchy_depth
        self.io_config = io_config
        self.multi_synapse = multi_synapse

        self.motif_configs = []# Tuple of tuples(node types, node i/o P(), connections, selection weight)

        if read_entire_population == False:
            print "gen population"
