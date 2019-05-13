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

file_location = '/home/adampcloth/Documents/Simulations/Motif/bandit/Champions/The data gathering with some gaps & failures/'
file_location = '/home/adampcloth/Documents/Simulations/Motif/logic/Champions/Huge batch of logic/Stochastic/'

key_words = ['stat', 'pl', 'max']

iterations = 200

# gather all file names
things_wanted = '*'
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

ticks = [i for i in range(iterations-1)]

plt.plot(ticks, mean_max_score_over_time)
# plt.plot(xfit, yfit, '-', color='gray')
mean_max_err_down = np.subtract(mean_max_score_over_time, stderr_max_score_over_time)
mean_max_err_up = np.add(mean_max_score_over_time, stderr_max_score_over_time)
plt.fill_between(ticks, mean_max_err_down, mean_max_err_up,
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.show()

plt.plot(ticks, mean_best_score_over_time)
# plt.plot(xfit, yfit, '-', color='gray')
mean_best_score_err_down = np.subtract(mean_best_score_over_time, stderr_best_score_over_time)
mean_best_score_err_up = np.add(mean_best_score_over_time, stderr_best_score_over_time)
plt.fill_between(ticks, mean_best_score_err_down, mean_best_score_err_up,
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.show()

plt.plot(ticks, mean_best_fitness_over_time)
# plt.plot(xfit, yfit, '-', color='gray')
mean_best_fitness_err_down = np.subtract(mean_best_fitness_over_time, stderr_best_fitness_over_time)
mean_best_fitness_err_up = np.add(mean_best_fitness_over_time, stderr_best_fitness_over_time)
plt.fill_between(ticks, mean_best_fitness_err_down, mean_best_fitness_err_up,
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.show()
# plt.xlim(0, 10);

print "done"

