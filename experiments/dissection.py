from methods.networks import motif_population
from methods.agents import agent_population
import numpy as np
import traceback
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing
import pathos.multiprocessing
import csv
from ast import literal_eval

#todo here will be where an agent and a motif population will be read in to examine what is present in a well performing agent

#todo could also do motif analysis and progression throught the simulation

# read motif populations over time
# see how each changes with time and how much weight it has

def motif_tracking(file_location, config):
    motifs_over_time = []
    weights_over_time = {}
    for i in range(3000):
        try:
            with open('{}/Motif population {}: {}'.format(file_location, i, config)) as motif_file:
                total_weight = 0
                weight_dist = []
                csvFile = csv.reader(motif_file)
                motif = False
                for row in csvFile:
                    temp = row
                    if temp[0] == 'node':
                        if motif:
                            try:
                                weights_over_time[motif['id']].append(motif['weight'])
                            except:
                                empty_slots = []
                                for j in range(i):
                                    empty_slots.append(0)
                                empty_slots.append(motif['weight'])
                                weights_over_time[motif['id']] = empty_slots
                            weight_dist.append([motif['id'], motif['weight']])
                            # self.insert_motif(deepcopy(motif), weight=motif['weight'], read=True)
                        motif = {}
                    atribute = temp[0]
                    del temp[0]
                    if atribute == 'depth':
                        temp = int(temp[0])
                    elif atribute == 'weight':
                        temp = literal_eval(temp[0])
                        total_weight += temp
                    elif atribute == 'conn' or atribute == 'io':
                        for j in range(len(temp)):
                            temp[j] = literal_eval(temp[j])
                    elif atribute == 'id':
                        temp = temp[0]
                    motif['{}'.format(atribute)] = temp
            for key in weights_over_time:
                if len(weights_over_time[key]) <= i:
                    weights_over_time[key].append(0)
            for j in range(len(weight_dist)):
                weight_dist[j][1] /= float(total_weight)
            weight_dist.sort(key=lambda x: x[1], reverse=True)
            motifs_over_time.append(weight_dist)
            print "finished ", i
        except:
            print "no file to read"
            break

    with open('{}/Motifs over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for iteration in motifs_over_time:
            writer.writerow(iteration)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

    with open('{}/5 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for motif in weights_over_time:
            data = [motif]
            entries = 0
            for weight in weights_over_time[motif]:
                if weight != 0:
                    entries += 1
                data.append(weight)
            if entries > 5:
                writer.writerow(data)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

    with open('{}/10 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for motif in weights_over_time:
            data = [motif]
            entries = 0
            for weight in weights_over_time[motif]:
                if weight != 0:
                    entries += 1
                data.append(weight)
            if entries > 10:
                writer.writerow(data)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

    with open('{}/1 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for motif in weights_over_time:
            data = [motif]
            entries = 0
            for weight in weights_over_time[motif]:
                if weight != 0:
                    entries += 1
                data.append(weight)
            if entries > 1:
                writer.writerow(data)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

    with open('{}/2 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for motif in weights_over_time:
            data = [motif]
            entries = 0
            for weight in weights_over_time[motif]:
                if weight != 0:
                    entries += 1
                data.append(weight)
            if entries > 2:
                writer.writerow(data)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

    with open('{}/20 Weights over time: {}.csv'.format(file_location, config), 'w') as weight_file:
        writer = csv.writer(weight_file, delimiter=',', lineterminator='\n')
        for motif in weights_over_time:
            data = [motif]
            entries = 0
            for weight in weights_over_time[motif]:
                if weight != 0:
                    entries += 1
                data.append(weight)
            if entries > 20:
                writer.writerow(data)
            # for motif in iteration:
            #     writer.writerow(motif)
        weight_file.close()

def read_motif(motif_id, iteration, file_location, config):
    motifs = motif_population(max_motif_size=3,
                              no_weight_bins=5,
                              no_delay_bins=5,
                              weight_range=(0.005, weight_max),
                              # delay_range=(2, 2.00001),
                              read_entire_population='{}/Motif population {}: {}.csv'.format(file_location, iteration, config),
                              population_size=200)

    motif = motifs.motif_configs[motif_id]
    motif_struct = motifs.read_motif(motif_id)
    # print motif_struct

file_location = 'runtime data/The start of screen and good results'

weight_max = 0.1
arm1 = 0.1
arm2 = 0.9
arm_len = 4
arms = []
for i in range(arm_len):
    arms.append([arm1, arm2])
    arms.append([arm2, arm1])
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
number_of_arms = 1
split = 1
reward_shape = True
reward = 0
noise_rate = 0
noise_weight = 0.01
maximum_depth = 7
size_fitness = False
spikes_fitness = False
random_arms = 0

config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}, w_max{}".format(
    reward_shape, reward, noise_rate, noise_weight, arms[0], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness, weight_max)

# motif_tracking(file_location, config)
read_motif('416623', 250, file_location, config)
