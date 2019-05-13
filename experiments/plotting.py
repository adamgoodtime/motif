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

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

class fom_data:
    def __init__(self,fom):
        self.fom_name = fom
        self.data = []
        self.colour = 'r'

    def add_data(self, dat):
        self.data.append(dat)

    def assign_colour(self, col):
        self.colour = col

class strat_data:
    def __init__(self,strat_name):
        self.strat_name = strat_name
        self.fom_data = []
        self.colour = 'g'

    def add_fom_data(self, fom_dat):
        self.fom_data.append(fom_dat)

    def assign_colour(self, col):
        self.colour = col

class ptn_experiments:
    def __init__(self,ptn_name):
        self.ptn_name = ptn_name
        self.strat_data = []

    def add_strat_data(self, str_dat):
        self.strat_data.append(str_dat)

def plot_energy_bit(ptn_exp):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = ptn_exp.ptn_name+": Bit Switching Energy Consumption"
    colours = ['b','g','r']
    styles = ['-','--']
    figdir = 'fjperbit'
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            for str_dat in ptn_exp.strat_data:
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Number of servers':
                        x_axis = np.array(fd.data[1:], dtype = np.integer)
                plt.xscale('log', basex=2)
                plt.xticks(x_axis)
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Total energy cons. per bit (fJ/bit)':
                        leg_entries.append(str_dat.strat_name)
                        x_data = np.array(fd.data[1:],dtype = np.float32)
                        if 'no_op' not in str_dat.strat_name:
                            if str_dat.strat_name == 'rand_path':
                                plt.plot(x_axis, x_data, color = str_dat.colour, marker='^',markersize=8, label = str_dat.strat_name)
                            else:
                                plt.plot(x_axis, x_data, color = str_dat.colour, linestyle='solid', label =str_dat.strat_name)
            plt.xlabel("nodes")
            plt.ylabel('Energy (fJ/bit)')
            ax = plt.gca()
            ax.set_xticklabels(x_axis)
            plt.legend()
            suptitle = title
            if 'butterfly' in suptitle:
                suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='.eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_makespan(ptn_exp):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = ptn_exp.ptn_name+": Makespan"
    colours = ['b','g','r']
    styles = ['-','--']
    figdir = 'makespan'
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            for str_dat in ptn_exp.strat_data:
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Number of servers':
                        x_axis = np.array(fd.data[1:], dtype = np.integer)
                plt.xscale('log', basex=2)
                plt.xticks(x_axis)
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Makespan':
                        leg_entries.append(str_dat.strat_name)
                        x_data = np.array(fd.data[1:],dtype = np.float32)
                        if 'no_op' not in str_dat.strat_name:
                            plt.plot(x_axis, x_data, color = str_dat.colour, linestyle='solid', label =str_dat.strat_name)
            plt.xlabel("nodes")
            plt.ylabel('makespan (microseconds)')
            ax = plt.gca()
            ax.set_xticklabels(x_axis)
            plt.legend()
            suptitle = title
            if 'butterfly' in suptitle:
                suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='.eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_latency(ptn_exp):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = ptn_exp.ptn_name+": latency"
    colours = ['b','g','r']
    styles = ['-','--']
    figdir = 'latency'
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            for str_dat in ptn_exp.strat_data:
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Number of servers':
                        x_axis = np.array(fd.data[1:], dtype = np.integer)
                plt.xscale('log', basex=2)
                plt.xticks(x_axis)
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Average flows latency':
                        leg_entries.append(str_dat.strat_name)
                        x_data = np.array(fd.data[1:],dtype = np.float32)
                        if 'no_op' not in str_dat.strat_name:
                            plt.plot(x_axis, x_data, color = str_dat.colour, linestyle='solid', label =str_dat.strat_name)
            plt.xlabel("nodes")
            plt.ylabel('Latency')
            ax = plt.gca()
            ax.set_xticklabels(x_axis)
            plt.legend()
            suptitle = title
            if 'butterfly' in suptitle:
                suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='.eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_iloss_breakdown(ptn_exp, case):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = ptn_exp.ptn_name+": "+case+" ILoss per flow."
    colours = ['b','g','r']
    styles = ['-','--']
    figdir = case+' iloss breakdown'
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            offset = 1
            width = 0.8
            for str_dat in ptn_exp.strat_data:
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Number of servers':
                        x_axis = np.array(fd.data[1:], dtype = np.integer)
                plt.xscale('log', basex=2)
                plt.xticks(x_axis)
                cr_fd = []
                wg_fd = []
                mzi_fd = []
                for fd in str_dat.fom_data:
                    if 'Insertion Loss' in fd.fom_name and 'wg' in fd.fom_name and case in fd.fom_name:
                        wg_fd = fd
                    if 'Insertion Loss' in fd.fom_name and 'cr' in fd.fom_name and case in fd.fom_name:
                        cr_fd = fd
                    if 'Insertion Loss' in fd.fom_name and 'mzi' in fd.fom_name and case in fd.fom_name:
                        mzi_fd = fd

                x_axis = np.logspace(4, 11, num=8, base=2)
                if 'max_state' in str_dat.strat_name :
                    offset=-np.diff(x_axis)/9
                if 'crossings' in str_dat.strat_name :
                    offset = 0
                # if 'rand_path' in str_dat.strat_name :
                #     offset=np.diff(x_axis)/9
                if 'state_changes' in str_dat.strat_name :
                    offset=np.diff(x_axis)/9
                if 'no_op' not in str_dat.strat_name and 'rand_path' not in str_dat.strat_name:
                    plt.bar(x_axis[:-1]+offset, np.array(wg_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, color = wg_fd.colour, align = 'center')
                    plt.bar(x_axis[:-1]+offset, np.array(cr_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, bottom = np.array(wg_fd.data[1:], dtype = np.float32), color = cr_fd.colour, align = 'center')
                    plt.bar(x_axis[:-1]+offset, np.array(mzi_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, bottom = np.array(wg_fd.data[1:], dtype = np.float32)+np.array(cr_fd.data[1:], dtype = np.float32), color = mzi_fd.colour, align = 'center')
                width = width*2
            plt.xlabel("nodes")
            plt.ylabel('Insertion Loss (dB)')
            bl_patch = mpatches.Patch(color='#4C5299', label='min_crossings')
            g_patch = mpatches.Patch(color='#4F994C', label='min_state_changes')
            r_patch = mpatches.Patch(color='#994C52', label='max_state_cross')
            # b_patch = mpatches.Patch(color='#000000', label='rand_path')
            ax = plt.gca()
            x_axis = np.array(x_axis, dtype = np.integer)
            ax.set_xticklabels(x_axis)
            # plt.legend()
            # plt.legend(handles=[r_patch, bl_patch, b_patch, g_patch])
            plt.legend(handles=[r_patch, bl_patch, g_patch])
            suptitle = title
            if 'butterfly' in suptitle:
                suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_total_iloss(ptn_exp):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = ptn_exp.ptn_name+": ILoss per flow."

    colours = ['b','g','r']
    styles = ['-','--']

    abs_total_iloss = [13.9702,19.1474,25.9246,35.9018,52.279,81.4562,136.2334]
    abs_total_iloss = np.array(abs_total_iloss, dtype = np.float32)
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs('total iloss')
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir('total iloss'):
                pass

        with cd('total iloss'):
            for str_dat in ptn_exp.strat_data:
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Number of servers':
                        x_axis = np.array(fd.data[1:], dtype = np.integer)
                plt.xscale('log', basex=2)
                plt.xticks(x_axis)
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Average flows Insertion Loss total':
                        leg_entries.append(str_dat.strat_name)
                        x_data = np.array(fd.data[1:],dtype = np.float32)
                        # fd.assign_colour('#7F3338')
                        # print(fd.data)
                        # print(str_dat.colour)
                        # if 'no_op' not in str_dat.strat_name:
                        if 'no_op' not in str_dat.strat_name and 'rand_path' not in str_dat.strat_name:
                            plt.plot(x_axis, x_data, color = str_dat.colour, linestyle='dashed', label ='avg_'+str_dat.strat_name)
                for fd in str_dat.fom_data:
                    if fd.fom_name == 'Max flows Insertion Loss total':
                        leg_entries.append(str_dat.strat_name)
                        x_data = np.array(fd.data[1:],dtype = np.float32)
                        # print(fd.data)
                        # print(str_dat.colour)
                        # if 'no_op' not in str_dat.strat_name:
                        if 'no_op' not in str_dat.strat_name and 'rand_path' not in str_dat.strat_name:
                            plt.plot(x_axis, x_data, color = str_dat.colour, linestyle='solid', label ='max_'+str_dat.strat_name)
            plt.xlabel("nodes")
            plt.ylabel('Insertion Loss (dB)')
            plt.plot(x_axis, abs_total_iloss, color = 'k', marker='^', linestyle = 'solid', label = 'abs_max_ILoss')
            ax = plt.gca()
            ax.set_xticklabels(x_axis)
            plt.legend()
            suptitle = title
            if 'butterfly' in suptitle:
                suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_iloss_breakdown_randompath(pattern_experiments):
    leg_entries = []
    x_labels = []
    x_axis = []
    title = "Max ILoss per flow."
    colours = ['b','g','r']
    styles = ['-','--']
    wg_col = '#000000'
    cr_col = '#000000'
    mzi_col = '#000000'
    figdir = 'Max Iloss breakdown randomPath'
    case = 'Max'
    workloads = ['randomapp', 'bisection', 'hotregion']
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            offset = 1
            width = 0.8
            for ptn_exp in pattern_experiments:
                if ptn_exp.ptn_name in workloads:
                    for str_dat in ptn_exp.strat_data:
                        if 'rand_path' in str_dat.strat_name:
                            for fd in str_dat.fom_data:
                                if fd.fom_name == 'Number of servers':
                                    x_axis = np.array(fd.data[1:], dtype = np.integer)
                            plt.xscale('log', basex=2)
                            plt.xticks(x_axis)
                            cr_fd = []
                            wg_fd = []
                            mzi_fd = []
                            for fd in str_dat.fom_data:
                                if 'Insertion Loss' in fd.fom_name and 'wg' in fd.fom_name and case in fd.fom_name:
                                    wg_fd = fd
                                    if 'randomapp' in ptn_exp.ptn_name :
                                        wg_col = '#191F66'
                                    if 'bisection' in ptn_exp.ptn_name:
                                        wg_col = '#1C6619'
                                    if 'hotregion' in ptn_exp.ptn_name:
                                        wg_col = '#66191F'
                                if 'Insertion Loss' in fd.fom_name and 'cr' in fd.fom_name and case in fd.fom_name:
                                    cr_fd = fd
                                    if 'randomapp' in ptn_exp.ptn_name :
                                        cr_col = '#4C5299'
                                    if 'bisection' in ptn_exp.ptn_name:
                                        cr_col = '#4F994C'
                                    if 'hotregion' in ptn_exp.ptn_name:
                                        cr_col = '#994C52'
                                if 'Insertion Loss' in fd.fom_name and 'mzi' in fd.fom_name and case in fd.fom_name:
                                    mzi_fd = fd
                                    if 'randomapp' in ptn_exp.ptn_name :
                                        mzi_col = '#7F85CC'
                                    if 'bisection' in ptn_exp.ptn_name:
                                        mzi_col = '#82CC7F'
                                    if 'hotregion' in ptn_exp.ptn_name:
                                        mzi_col = '#CC7F85'
                            # print(x_axis)
                            x_axis = np.logspace(4, 11, num=8, base=2)
                            if 'bisection' in ptn_exp.ptn_name:
                                offset=-np.diff(x_axis)/9
                            if 'randomapp' in ptn_exp.ptn_name :
                                offset = 0
                            if 'hotregion' in ptn_exp.ptn_name:
                                offset=np.diff(x_axis)/9
                            # if 'rand_path' in str_dat.strat_name :
                            #     offset=np.diff(x_axis)/9
                            # if 'no_op' not in str_dat.strat_name and 'rand_path' not in str_dat.strat_name:
                            plt.bar(x_axis[:-1]+offset, np.array(wg_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, color = wg_col, align = 'center')
                            plt.bar(x_axis[:-1]+offset, np.array(cr_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, bottom = np.array(wg_fd.data[1:], dtype = np.float32), color =cr_col, align = 'center')
                            plt.bar(x_axis[:-1]+offset, np.array(mzi_fd.data[1:], dtype = np.float32), width = np.diff(x_axis)/9, bottom = np.array(wg_fd.data[1:], dtype = np.float32)+np.array(cr_fd.data[1:], dtype = np.float32), color = mzi_col, align = 'center')
                            width = width*2
            plt.xlabel("nodes")
            plt.ylabel('Insertion Loss (dB)')
            bl_patch = mpatches.Patch(color='#4C5299', label='randomapp')
            g_patch = mpatches.Patch(color='#4F994C', label='bisection')
            r_patch = mpatches.Patch(color='#994C52', label='hotregion')
            # b_patch = mpatches.Patch(color='#000000', label='rand_path')
            ax = plt.gca()
            x_axis = np.array(x_axis, dtype = np.integer)
            ax.set_xticklabels(x_axis)
            # plt.legend()
            # plt.legend(handles=[r_patch, bl_patch, b_patch, g_patch])
            plt.legend(handles=[r_patch, bl_patch, g_patch])
            suptitle = title
            # if 'butterfly' in suptitle:
            #     suptitle = suptitle.replace('butterfly', 'AllReduce')
            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def plot_energy_bit_randomPath(pattern_experiments):
    leg_entries = []
    x_labels = []
    x_axis = []
    title ="Bit Switching Energy Consumption"
    colours = ['b','g','r']
    styles = ['-','--']
    figdir = 'fjperbit randomPath'
    workloads = ['randomapp', 'bisection', 'hotregion']
    col = 'k'
    mark = '-'
    with cd(os.path.join(os.getcwd(),'../figures')):
        try:
            os.makedirs(figdir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(figdir):
                pass

        with cd(figdir):
            for ptn_exp in pattern_experiments:
                if ptn_exp.ptn_name in workloads:
                    for str_dat in ptn_exp.strat_data:
                        if 'rand_path' in str_dat.strat_name:
                            for fd in str_dat.fom_data:
                                if fd.fom_name == 'Number of servers':
                                    x_axis = np.array(fd.data[1:], dtype = np.integer)
                            plt.xscale('log', basex=2)
                            plt.xticks(x_axis)
                            for fd in str_dat.fom_data:
                                if fd.fom_name == 'Total energy cons. per bit (fJ/bit)':
                                    leg_entries.append(ptn_exp.ptn_name)
                                    x_data = np.array(fd.data[1:],dtype = np.float32)
                                    if 'randomapp' in ptn_exp.ptn_name :
                                        col = 'b'
                                        mark = '^'
                                    if 'bisection' in ptn_exp.ptn_name:
                                        col = 'g'
                                        mark = 'v'
                                    if 'hotregion' in ptn_exp.ptn_name:
                                        col = 'r'
                                        mark = 's'
                                    plt.plot(x_axis, x_data, color = col, linestyle='solid', marker = mark, markersize=6, label =ptn_exp.ptn_name)

            plt.xlabel("nodes")
            plt.ylabel('Energy (fJ/bit)')
            ax = plt.gca()
            ax.set_xticklabels(x_axis)
            plt.legend()
            suptitle = title

            plt.suptitle(suptitle, fontsize=14)
            title = title.replace(":","")
            title+='.eps'
            plt.savefig(title.replace(" ","-"), format='eps', bbox_inches='tight')
            plt.clf()

def assign_colours(ptn_exp):
    #blues: #191F66, #33387F, #4C5299
    #greens: #1C6619, #367F33, #4F994C
    #reds: #66191F, #7F3338, #994C52
    for strat in ptn_exp.strat_data:
        for fom in strat.fom_data:
            if 'Insertion Loss' in fom.fom_name:
                # print(fom.fom_name)
                if 'total' in fom.fom_name:
                    # print(strat.strat_name)
                    if 'crossings' in strat.strat_name:
                        strat.assign_colour('b')
                    elif 'state_changes' in strat.strat_name:
                        strat.assign_colour('g')
                    elif 'max_state_cross' in strat.strat_name:
                        strat.assign_colour('r')
                    elif 'no_op' in strat.strat_name:
                        strat.assign_colour('m')
                    elif 'rand_path' in strat.strat_name:
                        strat.assign_colour('k')
                    else:
                        pass

                if 'wg' in fom.fom_name:
                    if 'crossings' in strat.strat_name:
                        fom.assign_colour('#191F66')
                    elif 'state_changes' in strat.strat_name:
                        fom.assign_colour('#1C6619')
                    elif 'max_state_cross' in strat.strat_name:
                        fom.assign_colour('#66191F')
                    elif 'rand_path' in strat.strat_name:
                        fom.assign_colour('#000000')
                    else:
                        pass
                if 'cr' in fom.fom_name:
                    if 'crossings' in strat.strat_name:
                        fom.assign_colour('#4C5299')
                    elif 'state_changes' in strat.strat_name:
                        fom.assign_colour('#4F994C')
                    elif 'max_state_cross' in strat.strat_name:
                        fom.assign_colour('#994C52')
                    elif 'rand_path' in strat.strat_name:
                        fom.assign_colour('#333333')
                    else:
                        pass

                if 'mzi' in fom.fom_name:
                    if 'crossings' in strat.strat_name:
                        fom.assign_colour('#7F85CC')
                    elif 'state_changes' in strat.strat_name:
                        fom.assign_colour('#82CC7F')
                    elif 'max_state_cross' in strat.strat_name:
                        fom.assign_colour('#CC7F85')
                    elif 'rand_path' in strat.strat_name:
                        fom.assign_colour('#999999')
                    else:
                        pass
            else:
                if 'crossings' in strat.strat_name:
                    strat.assign_colour('b')
                elif 'state_changes' in strat.strat_name:
                    strat.assign_colour('g')
                elif 'max_state_cross' in strat.strat_name:
                    strat.assign_colour('r')
                elif 'no_op' in strat.strat_name:
                    strat.assign_colour('m')
                elif 'rand_path' in strat.strat_name:
                    strat.assign_colour('k')
                else:
                    pass

def generate_batch_files(ptn_exp):
    abs_total_iloss = np.array([13.9702,19.1474,25.9246,35.9018,52.279,81.4562,136.2334])
    reduction = []
    energyReduction = []
    rand_path_energy = []
    try:
        os.makedirs('batches')
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir('batches'):
            pass
    with cd('batches'):
        with open(ptn_exp.ptn_name+"-reduction_over_abs_max", "a+") as reductionFile:
            redwr = csv.writer(reductionFile)
            redwr.writerow(abs_total_iloss)
            for strat in ptn_exp.strat_data:
                for fom in strat.fom_data:
                    snipstring = '()./'
                    for char in snipstring:
                        fom_name_snipped = fom.fom_name.replace(char, "")
                    if fom_name_snipped == 'Average flows Insertion Loss' or fom_name_snipped == 'Max flows Insertion Loss':
                        fom_name_snipped+= ' total'
                        fom.fom_name+=' total'
                    if 'Max flows Insertion Loss total' in fom.fom_name:
                        reduction = (1 - np.array(fom.data, dtype = np.float32)/abs_total_iloss)*100
                        red_data = reduction.tolist()
                        red_data.insert(0,strat.strat_name)
                        redwr.writerow(red_data)
                    out_data = []
                    out_fname = str(ptn_exp.ptn_name)+'-'+str(fom_name_snipped)
                    out_data = fom.data
                    out_data.insert(0,strat.strat_name)
                    with open(out_fname, 'a+') as myfile:
                        wr = csv.writer(myfile)
                        wr.writerow(out_data)
        with open(ptn_exp.ptn_name+"-energy_reduction_over_rand_path", "a+") as energyReductionFile:
            enredwr = csv.writer(energyReductionFile)
            for strat in ptn_exp.strat_data:
                for fom in strat.fom_data:
                    if 'Total energy cons. per bit' in fom.fom_name:
                        for strat_rand in ptn_exp.strat_data:
                            if 'rand_path' in strat_rand.strat_name:
                                for energyFom in strat_rand.fom_data:
                                    if 'Total energy cons. per bit' in energyFom.fom_name:
                                        rand_path_energy = np.array(energyFom.data[1:], dtype = np.float32)
                        energyReduction = (1 - np.array(fom.data[1:], dtype = np.float32)/rand_path_energy)*100
                        energyReduction = np.around(energyReduction, decimals = 3)
                        red_data = energyReduction.tolist()
                        red_data.insert(0,strat.strat_name)
                        enredwr.writerow(red_data)

def harvest_data(str_dat, fom_names, max_benes_degree):
    # eomzi_16_source_fr0.00_injmode0_wl-wload-16_seed123_lb0_mode0.list_applications

    for fom_name in fom_names:
        fd = fom_data(fom_name)
        for benes_degree in range(4,max_benes_degree+1):
            if fom_name == 'Makespan':
                out_fname = 'eomzi_'+str(2**benes_degree)+'_source_fr0.00_injmode0_wl-wload-'+str(2**benes_degree)+'_seed123_lb0_mode0.scheduling'
            else:
                out_fname = 'eomzi_'+str(2**benes_degree)+'_source_fr0.00_injmode0_wl-wload-'+str(2**benes_degree)+'_seed123_lb0_mode0.list_applications'
            try:
                with open(os.path.join(os.getcwd(),out_fname),'r') as out:
                    for line in out:
                        line = line.split(":")
                        line[-1] = line[-1].replace(" ","")
                        line[-1] = line[-1].replace("\n","")
                        if fom_name in line:
                            fd.add_data(line[-1])
            except IOError as e:
                print(e)
                fd.add_data(-1)
        str_dat.add_fom_data(fd)

def get_immediate_subdirectories_nobatches(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and 'batches' not in name]

def main():
    parser = argparse.ArgumentParser(prog="harvester.py", description = "Data harvester and plotter for phinrflow_eomzi simulations.", formatter_class= RawTextHelpFormatter)
    parser.add_argument("--max_benes_degree", help= "Max benes degree", default = 10)
    parser.add_argument("--dir", help= "Directory for experiments", default = 'experiments')
    parser.add_argument("--mode", help= "Which plots to generate. Choices: total_iloss, avg_iloss_breakdown, max_iloss_breakdown, fjperbit, latency, makespan, all, randompath", default = 'all')

    args = parser.parse_args()
    max_benes_degree = int(args.max_benes_degree)
    expdir = str(args.dir)
    mode = args.mode
    if mode not in ['total_iloss', 'avg_iloss_breakdown', 'max_iloss_breakdown', 'fjperbit', 'latency', 'makespan', 'all', 'randompath']:
        print('Mode specified incorrectly! Defaulting to all.')
        all_strategies=['min_crossings','min_state_changes','max_state_cross','rand_path', 'no_op']
    fom_names = ['Makespan','Number of servers','Average flows latency', 'Average flows Insertion Loss','Average flows Insertion Loss wg','Average flows Insertion Loss cr','Average flows Insertion Loss mzi','Max flows Insertion Loss','Max flows Insertion Loss wg','Max flows Insertion Loss cr','Max flows Insertion Loss mzi','Maximum power cons. (mW)','Total energy cons. per byte (fJ/byte)','Total energy cons. per bit (fJ/bit)']
    pattern_experiments = []
    try:
        os.makedirs('figures')
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir('figures'):
            pass

    with cd(expdir):
        for ptn_dir in os.listdir():
            with cd(ptn_dir):
                dirpath = os.getcwd()
                ptn_name = os.path.basename(dirpath)
                ptn_exp = ptn_experiments(ptn_name)
                print(ptn_name)
                if 'batches' in os.listdir():
                    shutil.rmtree('batches')

                for strat_dir in get_immediate_subdirectories_nobatches(os.getcwd()):
                    with cd(strat_dir):
                        strat_name = os.path.basename(os.getcwd())
                        st_dat = strat_data(strat_name)
                        harvest_data(st_dat, fom_names, max_benes_degree)
                    ptn_exp.add_strat_data(st_dat)
                generate_batch_files(ptn_exp)
            assign_colours(ptn_exp)
            if mode == 'total_iloss' or mode == 'all':
                plot_total_iloss(ptn_exp)
            if mode == 'avg_iloss_breakdown' or mode == 'all':
                plot_iloss_breakdown(ptn_exp, 'Average')
            if mode == 'max_iloss_breakdown' or mode == 'all':
                plot_iloss_breakdown(ptn_exp, 'Max')
            if mode == 'fjperbit' or mode == 'all':
                plot_energy_bit(ptn_exp)
            if mode == 'makespan' or mode == 'all':
                plot_makespan(ptn_exp)
            if mode == 'latency' or mode == 'all':
                plot_latency(ptn_exp)
            pattern_experiments.append(ptn_exp)
            if mode == 'randompath':
                plot_iloss_breakdown_randompath(pattern_experiments)
                plot_energy_bit_randomPath(pattern_experiments)
    # colours = ['b','g','r','c','m','y','k','w']

if __name__ == "__main__":
  main()
