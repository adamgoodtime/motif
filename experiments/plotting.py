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

def plot_tracking_of_scores(file_location, key_words, iterations):
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

    plt.plot(ticks, mean_max_score_over_time, 'r')
    # plt.plot(xfit, yfit, '-', color='gray')
    mean_max_err_down = np.subtract(mean_max_score_over_time, stderr_max_score_over_time)
    mean_max_err_up = np.add(mean_max_score_over_time, stderr_max_score_over_time)
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
    mean_best_score_err_down = np.subtract(mean_best_score_over_time, stderr_best_score_over_time)
    mean_best_score_err_up = np.add(mean_best_score_over_time, stderr_best_score_over_time)
    plt.fill_between(ticks, mean_best_score_err_down, mean_best_score_err_up,
    # plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                     color='green', alpha=0.2)
    # plt.show()

    plt.plot(ticks, mean_best_fitness_over_time, 'k')
    # plt.plot(xfit, yfit, '-', color='gray')
    mean_best_fitness_err_down = np.subtract(mean_best_fitness_over_time, stderr_best_fitness_over_time)
    mean_best_fitness_err_up = np.add(mean_best_fitness_over_time, stderr_best_fitness_over_time)
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

def combine_graphs(data_list):
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
            plt.suptitle(title, fontsize=14)
            plt.legend(loc='lower right')
            plt.fill_between(ticks, data_dict[data_type][setting][1], data_dict[data_type][setting][2], color=colour[1], alpha=0.2)
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

    for heading in headings:
        plt.clf()
        colour = get_colours()
        plt.plot(ticks, mean_over_time[heading], colour[0], label=heading)
        title = 'XOR {}'.format(heading)
        plt.suptitle(title, fontsize=14)
        plt.legend(loc='lower right')
        up = np.add(mean_over_time[heading], stderr_over_time[heading])
        down = np.subtract(mean_over_time[heading], stderr_over_time[heading])
        plt.fill_between(ticks, up, down, color=colour[1], alpha=0.2)
        plt.show()


file_location = '/home/adampcloth/Documents/Simulations/Motif/bandit/Champions/The data gathering with some gaps & failures/'
file_location = '/home/adampcloth/Documents/Simulations/Motif/logic/Champions/Huge batch of logic/Non stoc 25 const delay/'

key_words_no_pl_dev = ['5 ave', 'dev_n']
key_words_pl_dev = ['pl', 'dev_n']
key_words_no_pl_no_dev = ['5 ave', '.0 o']
key_words_pl_no_dev = ['pl', '.0 o']

key_words_list = {}
key_words_list['no_pl_dev'] = key_words_no_pl_dev
key_words_list['pl_dev'] = key_words_pl_dev
key_words_list['no_pl_no_dev'] = key_words_no_pl_no_dev
key_words_list['pl_no_dev'] = key_words_pl_no_dev

iterations = 200
ticks = [i for i in range(iterations - 1)]

# data_list = {}
# for key_words in key_words_list:
#     data_list[key_words] = plot_tracking_of_scores(file_location, key_words_list[key_words], iterations)
#
# colour_list = [['g', 'green'], ['r', 'red'], ['b', 'blue'], ['k', 'black']]
# combine_graphs(data_list)

tracking_list = {}
for key_words in key_words_list:
    tracking_list[key_words] = plot_tracking_of_networks(file_location, key_words_list[key_words], iterations, 'best score ')

print "done"

