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
    print '{}/Motif population {}: {}'.format(file_location, 0, config)
    for i in range(3000):
        try:
            with open('{}/Motif population {}: {}.csv'.format(file_location, i, config)) as motif_file:
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

def mutate_anal(file_location, config):
    print "attempting file - {}/mutate keys for {}: {}.csv".format(file_location, 0, config)
    mutate_track = {}
    mutate_track['motif'] = []
    mutate_track['new'] = []
    mutate_track['node'] = []
    mutate_track['io'] = []
    mutate_track['m_add'] = []
    mutate_track['m_gone'] = []
    mutate_track['c_add'] = []
    mutate_track['c_gone'] = []
    mutate_track['param_w'] = []
    mutate_track['param_d'] = []
    mutate_track['sex'] = []
    mutate_track['asex'] = []
    for i in range(3000):
        try:
            with open('{}/mutate keys for {}: {}.csv'.format(file_location, i, config)) as mutate_file:
                csvFile = csv.reader(mutate_file)
                print "reading iteration", i
                for row in csvFile:
                    temp = row
                    if temp[0] == 'motif':
                        motif = literal_eval(temp[1])
                    elif temp[0] == 'new':
                        new = literal_eval(temp[1])
                    elif temp[0] == 'node':
                        node = literal_eval(temp[1])
                    elif temp[0] == 'io':
                        io = literal_eval(temp[1])
                    elif temp[0] == 'm_add':
                        m_add = literal_eval(temp[1])
                    elif temp[0] == 'm_gone':
                        m_gone = literal_eval(temp[1])
                    elif temp[0] == 'c_add':
                        c_add = literal_eval(temp[1])
                    elif temp[0] == 'c_gone':
                        c_gone = literal_eval(temp[1])
                    elif temp[0] == 'param_w':
                        param_w = literal_eval(temp[1])
                    elif temp[0] == 'param_d':
                        param_d = literal_eval(temp[1])
                    elif temp[0] == 'mum':
                        mum = literal_eval(temp[1])
                        mum_fitness = mum[0]
                        mum_score = mum[1]
                    elif temp[0] == 'dad':
                        dad = literal_eval(temp[1])
                        dad_fitness = dad[0]
                        dad_score = dad[1]
                    elif temp[0] == 'sex':
                        sex = literal_eval(temp[1])
                    else:
                        if i > 1:
                            mum_fitness_change = agent_fitness - mum_fitness
                            dad_fitness_change = agent_fitness - dad_fitness
                            ave_fitness_change = (mum_fitness_change + dad_fitness_change) / 2
                            mum_score_change = agent_score - mum_score
                            dad_score_change = agent_score - dad_score
                            ave_score_change = (mum_score_change + dad_score_change) / 2
                            # if motif:
                            #     mutate_track['motif'].append([ave_fitness_change * motif, ave_score_change * motif])
                            # if new:
                            #     mutate_track['new'].append([ave_fitness_change * new, ave_score_change * new])
                            # if node:
                            #     mutate_track['node'].append([ave_fitness_change * node, ave_score_change * node])
                            # if io:
                            #     mutate_track['io'].append([ave_fitness_change * io, ave_score_change * io])
                            # if m_add:
                            #     mutate_track['m_add'].append([ave_fitness_change * m_add, ave_score_change * m_add])
                            # if m_gone:
                            #     mutate_track['m_gone'].append([ave_fitness_change * m_gone, ave_score_change * m_gone])
                            # if c_add:
                            #     mutate_track['c_add'].append([ave_fitness_change * c_add, ave_score_change * c_add])
                            # if c_gone:
                            #     mutate_track['c_gone'].append([ave_fitness_change * c_gone, ave_score_change * c_gone])
                            # if param_w:
                            #     mutate_track['param_w'].append([ave_fitness_change * param_w, ave_score_change * param_w])
                            # if param_d:
                            #     mutate_track['param_d'].append([ave_fitness_change * param_d, ave_score_change * param_d])
                            # if sex:
                            #     mutate_track['sex'].append([ave_fitness_change, ave_score_change])
                            # if not sex:
                            #     mutate_track['asex'].append([ave_fitness_change, ave_score_change])
                            if motif:
                                mutate_track['motif'].append(ave_score_change * motif)
                            if new:
                                mutate_track['new'].append(ave_score_change * new)
                            if node:
                                mutate_track['node'].append(ave_score_change * node)
                            if io:
                                mutate_track['io'].append(ave_score_change * io)
                            if m_add:
                                mutate_track['m_add'].append(ave_score_change * m_add)
                            if m_gone:
                                mutate_track['m_gone'].append(ave_score_change * m_gone)
                            if c_add:
                                mutate_track['c_add'].append(ave_score_change * c_add)
                            if c_gone:
                                mutate_track['c_gone'].append(ave_score_change * c_gone)
                            if param_w:
                                mutate_track['param_w'].append(ave_score_change * param_w)
                            if param_d:
                                mutate_track['param_d'].append(ave_score_change * param_d)
                            if sex:
                                mutate_track['sex'].append(ave_score_change)
                            if not sex:
                                mutate_track['asex'].append(ave_score_change)
                            if motif:
                                mutate_track['motif'].append(ave_fitness_change * motif)
                            if new:
                                mutate_track['new'].append(ave_fitness_change * new)
                            if node:
                                mutate_track['node'].append(ave_fitness_change * node)
                            if io:
                                mutate_track['io'].append(ave_fitness_change * io)
                            if m_add:
                                mutate_track['m_add'].append(ave_fitness_change * m_add)
                            if m_gone:
                                mutate_track['m_gone'].append(ave_fitness_change * m_gone)
                            if c_add:
                                mutate_track['c_add'].append(ave_fitness_change * c_add)
                            if c_gone:
                                mutate_track['c_gone'].append(ave_fitness_change * c_gone)
                            if param_w:
                                mutate_track['param_w'].append(ave_fitness_change * param_w)
                            if param_d:
                                mutate_track['param_d'].append(ave_fitness_change * param_d)
                            if sex:
                                mutate_track['sex'].append(ave_fitness_change)
                            if not sex:
                                mutate_track['asex'].append(ave_fitness_change)
                            motif = 0
                            nen = 0
                            io = 0
                            m_add = 0
                            m_gone = 0
                            c_add = 0
                            c_gone = 0
                            param_w = 0
                            param_d = 0
                            mum_fitness = 0
                            mum_score = 0
                            dad_fitness = 0
                            dad_score = 0
                            sex = 0
                        agent_fitness = literal_eval(temp[1])
                        agent_score = literal_eval(temp[2])
        except:
            print "no more files or bad config"
            if i > 10:
                with open('{}/Mutate over time: {}.csv'.format(file_location, config), 'w') as mutate_file:
                    writer = csv.writer(mutate_file, delimiter=',', lineterminator='\n')
                    row = []
                    for operator in mutate_track:
                        row.append('{} scr'.format(operator))
                        row.append('{} fit'.format(operator))
                    writer.writerow(row)
                    row = []
                    i = 0
                    while True:
                        row = []
                        all_done = True
                        for operator in mutate_track:
                            if len(mutate_track[operator]) > i:
                                all_done = False
                                row.append(mutate_track[operator][i])
                                row.append(mutate_track[operator][i+1])
                            else:
                                row.append('')
                        writer.writerow(row)
                        if all_done:
                            break
                        i += 2
                    mutate_file.close()
            break

file_location = 'runtime data/The start of screen and good results'
file_location = 'runtime data/spalloc fails with really good 0.6'

weight_max = 0.1
arm1 = 0.7
arm2 = 0.3
arm_len = 4
arms = []
for i in range(arm_len):
    arms.append([arm1, arm2])
    arms.append([arm2, arm1])
# arms = [[0.4, 0.6], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]
number_of_arms = 1
split = 1
reward_shape = False
reward = 0
noise_rate = 0
noise_weight = 0.01
maximum_depth = 5
size_fitness = False
spikes_fitness = False
random_arms = 0
viable_parents = 1

# mutate keys for 0: bandit reward_shape:False, reward:0, noise r-w:0-0.01, arms:[0.6, 0.4]-8-0, max_d5, size:False, spikes:False, w_max0.1, rents0.01.csv
# mutate keys for 33: bandit reward_shape:False, reward:0, noise r-w:0-0.016789, arms:[0.4, 0.6]-8-0, max_d4, size:False, spikes:False, w_max0.1.csv
# mutate keys for 0: bandit reward_shape:False, reward:0, noise r-w:0-0.016789, arms:[0.4, 0.6]-8-0, max_d4, size:False, spikes:False, w_max0.1.csv
# mutate keys for 41: bandit reward_shape:False, reward:0, noise r-w:0-0.016789, arms:[0.4, 0.6]-8-0, max_d4, size:False, spikes:False, w_max0.1.csv
# runtime data/spalloc fails with really good 0.6/mutate keys for 34: bandit reward_shape:False, reward:0, noise r-w:0-0.01, arms:[0.7, 0.3]-8-0, max_d5, size:False, spikes:False, w_max0.1, rents1.csv

config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}, w_max{}, rents{}".format(
    reward_shape, reward, noise_rate, noise_weight, arms[0], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness, weight_max, viable_parents)
#
# config = "bandit reward_shape:{}, reward:{}, noise r-w:{}-{}, arms:{}-{}-{}, max_d{}, size:{}, spikes:{}, w_max{}".format(
#     reward_shape, reward, noise_rate, noise_weight, arms[0], len(arms), random_arms, maximum_depth, size_fitness, spikes_fitness, weight_max)

# motif_tracking(file_location, config)
# read_motif('152946', 96, file_location, config)
mutate_anal(file_location, config)