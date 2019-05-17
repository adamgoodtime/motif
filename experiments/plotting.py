import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import argparse
import errno
from argparse import RawTextHelpFormatter
from contextlib import contextmanager
import os.path
import shutil
import matplotlib.patches as mpatches
import glob
from ast import literal_eval

def plot_tracking_of_scores(file_location, key_words, iterations, plot=False):
    # gather all file names
    things_wanted = '*sta*'
    for thing in key_words:
        things_wanted += thing
        things_wanted += '*'

    list_of_files = glob.glob(file_location+things_wanted)

    print list_of_files

    # collect all csv data
    list_of_data = []
    for file_name in list_of_files:
        with open(file_name) as motif_file:
            csvFile = csv.reader(motif_file)
            file_data = [file_name]
            for row in csvFile:
                file_data.append(row)
            list_of_data.append(file_data)

    # extract the important data
    row_heading = 'empty'
    record = False
    all_the_data = []
    for test in list_of_data:
        data = {}
        for row in test:
            if not record:
                if row[0] == 'maximum score' or row[0] == 'best performance score' or row[0] == 'best performance fitness':
                    row_heading = row[0]
                    record = True
            else:
                data[row_heading] = map(float, row)
                record = False
        if len(data['maximum score']) < iterations:
            print "broken run of file ", test[0]
        else:
            all_the_data.append(data)

    # convert to a time based format
    max_score_over_time = []
    best_score_over_time = []
    best_fitness_over_time = []
    for i in range(iterations-1):
        max_score_at_time = []
        best_score_at_time = []
        best_fitness_at_time = []
        for data in all_the_data:
            max_score_at_time.append(data['maximum score'][i])
            best_score_at_time.append(data['best performance score'][i])
            best_fitness_at_time.append(data['best performance fitness'][i])
        max_score_over_time.append(max_score_at_time)
        best_score_over_time.append(best_score_at_time)
        best_fitness_over_time.append(best_fitness_at_time)

    mean_max_score_over_time = []
    mean_best_score_over_time = []
    mean_best_fitness_over_time = []
    stderr_max_score_over_time = []
    stderr_best_score_over_time = []
    stderr_best_fitness_over_time = []

    for i in range(iterations-1):
        mean_max_score_over_time.append(np.mean(max_score_over_time[i]))
        mean_best_score_over_time.append(np.mean(best_score_over_time[i]))
        mean_best_fitness_over_time.append(np.mean(best_fitness_over_time[i]))
        stderr_max_score_over_time.append(np.std(max_score_over_time[i]) / np.sqrt(len(all_the_data)))
        stderr_best_score_over_time.append(np.std(best_score_over_time[i]) / np.sqrt(len(all_the_data)))
        stderr_best_fitness_over_time.append(np.std(best_fitness_over_time[i]) / np.sqrt(len(all_the_data)))

    mean_max_err_down = np.subtract(mean_max_score_over_time, stderr_max_score_over_time)
    mean_max_err_up = np.add(mean_max_score_over_time, stderr_max_score_over_time)

    mean_best_score_err_down = np.subtract(mean_best_score_over_time, stderr_best_score_over_time)
    mean_best_score_err_up = np.add(mean_best_score_over_time, stderr_best_score_over_time)

    mean_best_fitness_err_down = np.subtract(mean_best_fitness_over_time, stderr_best_fitness_over_time)
    mean_best_fitness_err_up = np.add(mean_best_fitness_over_time, stderr_best_fitness_over_time)

    if plot:
        plt.plot(ticks, mean_max_score_over_time, 'r')
        # plt.plot(xfit, yfit, '-', color='gray')
        plt.fill_between(ticks, mean_max_err_down, mean_max_err_up,
        # plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                         color='red', alpha=0.2)
        # plt.show()
        # plt.xlabel("Iterations")
        # plt.ylabel('Score')
        # title = 'Xor problem'
        # plt.suptitle(title, fontsize=14)
        # title = title.replace(":","")
        # title+='.eps'
        # plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
        # plt.clf()

        plt.plot(ticks, mean_best_score_over_time, 'g')
        # plt.plot(xfit, yfit, '-', color='gray')
        plt.fill_between(ticks, mean_best_score_err_down, mean_best_score_err_up,
        # plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                         color='green', alpha=0.2)
        # plt.show()

        plt.plot(ticks, mean_best_fitness_over_time, 'k')
        # plt.plot(xfit, yfit, '-', color='gray')
        plt.fill_between(ticks, mean_best_fitness_err_down, mean_best_fitness_err_up,
        # plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                         color='gray', alpha=0.2)
    # plt.show()
    # plt.xlim(0, 10);

    data_dict = {}
    data_dict['mean_max_score_over_time'] = [mean_max_score_over_time, mean_max_err_down, mean_max_err_up]
    data_dict['mean_best_score_over_time'] = [mean_best_score_over_time, mean_best_score_err_down, mean_best_score_err_up]
    data_dict['mean_best_fitness_over_time'] = [mean_best_fitness_over_time, mean_best_fitness_err_down, mean_best_fitness_err_up]
    # data_dict['stderr_max_score_over_time'] = stderr_max_score_over_time
    # data_dict['stderr_best_score_over_time'] = stderr_best_score_over_time
    # data_dict['stderr_best_fitness_over_time'] = stderr_best_fitness_over_time

    return data_dict

def combine_graphs(data_list, save=False):
    # flip dimensions
    data_dict = {}
    data_dict['mean_max_score_over_time'] = {}
    data_dict['mean_best_score_over_time'] = {}
    data_dict['mean_best_fitness_over_time'] = {}
    # data_dict['stderr_max_score_over_time'] = {}
    # data_dict['stderr_best_score_over_time'] = {}
    # data_dict['stderr_best_fitness_over_time'] = {}
    for setting in data_list:
        for data_type in data_list[setting]:
            data_dict[data_type][setting] = data_list[setting][data_type]

    ticks = [i for i in range(iterations-1)]
    for data_type in data_dict:
        plt.clf()
        for setting in data_dict[data_type]:
            colour = get_colours()
            plt.plot(ticks, data_dict[data_type][setting][0], colour[0], label=setting)
            title = 'XOR {}'.format(data_type)
            title = title.replace('_', ' ')
            plt.suptitle(title, fontsize=14)
            # plt.legend(loc='lower right')
            plt.legend(loc='upper left')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.fill_between(ticks, data_dict[data_type][setting][1], data_dict[data_type][setting][2], color=colour[1], alpha=0.2)
        if save:
            title += '.svg'
            plt.savefig(title, format='svg', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

def get_colours():
    global colour_list
    try:
        colour = colour_list[0]
    except:
        colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
        colour = colour_list[0]
    del colour_list[0]
    return colour

def plot_tracking_of_networks(file_location, key_words, iterations, tracking):
    # gather all file names
    things_wanted = '*track*'
    for thing in key_words:
        things_wanted += thing
        things_wanted += '*'

    list_of_files = glob.glob(file_location+things_wanted)

    print list_of_files

    # collect all csv data
    list_of_data = []
    for file_name in list_of_files:
        with open(file_name) as motif_file:
            csvFile = csv.reader(motif_file)
            file_data = [file_name]
            for row in csvFile:
                file_data.append(row)
            list_of_data.append(file_data)

    # extract the important data
    row_heading = 'empty'
    all_the_data = []
    headings = []
    for test in list_of_data:
        data = {}
        for row in test:
            if tracking in row[0]:
                row_heading = row[0].replace(tracking, "")
                row_heading = row_heading.replace('\n', "")
                if row_heading not in headings:
                    headings.append(row_heading)
                data[row_heading] = literal_eval(row[1])
        # if len(data[row_heading]) < iterations:
        #     print "broken run of file ", test[0]
        all_the_data.append(data)

    # convert to a time based format
    tracking_dict = {}
    for heading in headings:
        tracking_dict[heading] = []
    for i in range(iterations-1):
        for heading in headings:
            tracking_at_time = []
            for data in all_the_data:
                tracking_at_time.append(data[heading][i])
            tracking_dict[heading].append(tracking_at_time)

    mean_over_time = {}
    stderr_over_time = {}
    for heading in headings:
        mean_over_time[heading] = []
        stderr_over_time[heading] = []

    for i in range(iterations-1):
        for heading in headings:
            mean_over_time[heading].append(np.mean(tracking_dict[heading][i]))
            stderr_over_time[heading].append(np.std(tracking_dict[heading][i]) / np.sqrt(len(all_the_data)))

    # for heading in headings:
    #     plt.clf()
    #     colour = get_colours()
    #     plt.plot(ticks, mean_over_time[heading], colour[0], label=heading)
    #     title = 'XOR {}'.format(heading)
    #     plt.suptitle(title, fontsize=14)
    #     plt.legend(loc='lower right')
    #     up = np.add(mean_over_time[heading], stderr_over_time[heading])
    #     down = np.subtract(mean_over_time[heading], stderr_over_time[heading])
    #     plt.fill_between(ticks, up, down, color=colour[1], alpha=0.2)
    #     plt.show()

    data_dict = {}
    for heading in headings:
        data_dict[heading] = [mean_over_time[heading], np.subtract(mean_over_time[heading], stderr_over_time[heading]),
                              np.add(mean_over_time[heading], stderr_over_time[heading])]
    return data_dict

def combine_tracking(data_list, save=False):
    # flip dimensions
    data_dict = {}
    for setting in data_list:
        for data_type in data_list[setting]:
            data_dict[data_type] = {}
    for setting in data_list:
        for data_type in data_list[setting]:
            data_dict[data_type][setting] = data_list[setting][data_type]

    ticks = [i for i in range(iterations-1)]
    for data_type in data_dict:
        plt.clf()
        for setting in data_dict[data_type]:
            colour = get_colours()
            plt.plot(ticks, data_dict[data_type][setting][0], colour[0], label=setting)
            title = 'XOR average {} over time'.format(convert_to_proper_label(data_type))
            plt.suptitle(title, fontsize=14)
            # plt.legend(loc='lower right')
            plt.legend(loc='upper left')
            plt.xlabel('Iteration')
            plt.ylabel(convert_to_proper_label(data_type))
            plt.fill_between(ticks, data_dict[data_type][setting][1], data_dict[data_type][setting][2], color=colour[1], alpha=0.2)
        if save:
            title += '.svg'
            plt.savefig(title, format='svg', bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

def convert_to_proper_label(label):
    proper_label = ''
    if label == 'pl':
        proper_label = 'plasticity ratio'
    elif label == 'conn':
        proper_label = 'number of connections'
    elif label == 'inhib':
        proper_label = 'number of inhibitory neurons'
    elif label == 'excite':
        proper_label = 'number of excitatory neurons'
    elif label == 'depth':
        proper_label = 'maximum depth of network'
    return proper_label

def plot_individual_runs(save=False):
    file_location = '/home/adampcloth/Documents/Simulations/Motif/bandit/Champions/The data gathering with some gaps & failures/'
    file_location = '/home/adampcloth/Documents/Simulations/Motif/logic/Champions/Huge batch of logic/Non stoc 25 max delay/'
    key_words_no_pl_dev = ['5 ave', 'dev_n']
    key_words_pl_dev = ['pl', 'dev_n']
    key_words_no_pl_no_dev = ['5 ave', '.0 o']
    key_words_pl_no_dev = ['pl', '.0 o']

    key_words_list = {}
    key_words_list['non plastic with neuron development'] = key_words_no_pl_dev
    key_words_list['plastic with neuron development'] = key_words_pl_dev
    key_words_list['non plastic no neuron development'] = key_words_no_pl_no_dev
    key_words_list['plastic no neuron development'] = key_words_pl_no_dev

    data_list = {}
    for key_words in key_words_list:
        data_list[key_words] = plot_tracking_of_scores(file_location, key_words_list[key_words], iterations)

    colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
    combine_graphs(data_list, save)

    tracking_list = {}
    for key_words in key_words_list:
        tracking_list[key_words] = plot_tracking_of_networks(file_location, key_words_list[key_words], iterations, 'best score ')

    combine_tracking(tracking_list, save)

def plot_combined_runs(save=False):
    top_directory = '/home/adampcloth/Documents/Simulations/Motif/logic/Champions/Huge batch of logic/'
    file_location1 = 'Non stoc 25 max delay'
    file_location2 = 'Non stoc 25 const delay'
    file_location3 = 'Non stoc 50 max delay'
    file_locations = {file_location1: top_directory+file_location1+'/',
                      file_location2: top_directory+file_location2+'/',
                      file_location3: top_directory+file_location3+'/'}
    key_words_no_pl_dev = ['5 ave', 'dev_n']
    key_words_pl_dev = ['pl', 'dev_n']
    key_words_no_pl_no_dev = ['5 ave', '.0 o']
    key_words_pl_no_dev = ['pl', '.0 o']

    key_words_list = {}
    key_words_list['non plastic with neuron development'] = key_words_no_pl_dev
    key_words_list['plastic with neuron development'] = key_words_pl_dev
    key_words_list['non plastic without neuron development'] = key_words_no_pl_no_dev
    key_words_list['plastic without neuron development'] = key_words_pl_no_dev
    ticks = [i for i in range(iterations - 1)]

    all_data_list = {}
    all_tracking_list = {}
    for key_words in key_words_list:
        all_data_list[key_words] = {}
        all_tracking_list[key_words] = {}
        for file_location in file_locations:
            all_data_list[key_words][file_location] = []
            all_tracking_list[key_words][file_location] = []

    for file_location in file_locations:
        data_list = {}
        for key_words in key_words_list:
            data_list[key_words] = plot_tracking_of_scores(file_locations[file_location], key_words_list[key_words], iterations)
            all_data_list[key_words][file_location].append(data_list[key_words])

    plot_different_scores(all_data_list)

    # for file_location in file_locations:
    #     tracking_list = {}
    #     for key_words in key_words_list:
    #         tracking_list[key_words] = plot_tracking_of_networks(file_locations[file_location], key_words_list[key_words], iterations, 'best score ')
    #         all_tracking_list[key_words][file_location].append(tracking_list[key_words])
    #
    # plot_different_trackings(all_tracking_list)

def plot_different_scores(data_list, save=False):
    global colour_list
    # flip dimensions
    # currently config, test, metric
    # want config, metric, test
    # data_dict = {}
    # data_dict['mean_max_score_over_time'] = {}
    # data_dict['mean_best_score_over_time'] = {}
    # data_dict['mean_best_fitness_over_time'] = {}
    data_dict = {}
    for config in data_list:
        data_dict[config] = {}
        temp_dict = {}
        first = True
        for setting in data_list[config]:
            for metric in data_list[config][setting][0]:
                if first:
                    temp_dict[metric] = {}
                temp_dict[metric][setting] = []
            first = False
        data_dict[config] = temp_dict

    for config in data_list:
        for setting in data_list[config]:
            for metric in data_list[config][setting][0]:
                data_dict[config][metric][setting] = data_list[config][setting][0][metric]

    for config in data_dict:
        plt.clf()
        for metric in data_dict[config]:
            colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
            for setting in data_dict[config][metric]:
                colour = get_colours()
                plt.plot(ticks, data_dict[config][metric][setting][0], colour[0], label=setting)
                title = 'XOR {} for {}'.format(metric, config)
                title = title.replace('_', ' ')
                plt.suptitle(title, fontsize=14)
                plt.legend(loc='lower right')
                plt.xlabel('Iteration')
                plt.ylabel('Score')
                plt.fill_between(ticks, data_dict[config][metric][setting][1], data_dict[config][metric][setting][2], color=colour[1], alpha=0.2)
            if save:
                title += '.svg'
                plt.savefig(title, format='svg', bbox_inches='tight')
                plt.clf()
            else:
                plt.show()

def plot_different_trackings(data_list, save=False):
    global colour_list
    # flip dimensions
    data_dict = {}
    for config in data_list:
        data_dict[config] = {}
        temp_dict = {}
        first = True
        for setting in data_list[config]:
            for metric in data_list[config][setting][0]:
                if first:
                    temp_dict[metric] = {}
                temp_dict[metric][setting] = []
            first = False
        data_dict[config] = temp_dict

    for config in data_list:
        for setting in data_list[config]:
            for metric in data_list[config][setting][0]:
                data_dict[config][metric][setting] = data_list[config][setting][0][metric]

    for config in data_dict:
        plt.clf()
        for metric in data_dict[config]:
            colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
            for setting in data_dict[config][metric]:
                colour = get_colours()
                plt.plot(ticks, data_dict[config][metric][setting][0], colour[0], label=setting)
                title = 'XOR average {} over time for {}'.format(convert_to_proper_label(metric), config)
                plt.suptitle(title, fontsize=14)
                plt.legend(loc='lower right')
                plt.xlabel('Iteration')
                plt.ylabel(convert_to_proper_label(metric))
                plt.fill_between(ticks, data_dict[config][metric][setting][1], data_dict[config][metric][setting][2], color=colour[1], alpha=0.2)
            if save:
                title += '.svg'
                plt.savefig(title, format='svg', bbox_inches='tight')
                plt.clf()
            else:
                plt.show()


colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
iterations = 200
ticks = [i for i in range(iterations - 1)]

plot_individual_runs(True)
# plot_combined_runs()

print "done"

