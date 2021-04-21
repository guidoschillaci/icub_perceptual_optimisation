import imageio
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import os
from scipy import stats
import tabulate
import tkinter
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

# sthis only for correcting issue with 11 elements instead of 10 in marker det tests
start_marker_det_index = 1 # 1 + 10 epochs?

def make_figure(means, stddevs, title, xlabel, ylabel, y_lim):
    fig1 = plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(y_lim)
    plt.errorbar(range(len(means)), means, stddevs, capsize=5)#, errorevery=499)
    filename = title +'.jpg'
    plt.savefig(filename)
    plt.close()

def make_figure_loss(means_l, stddevs_l, means_val, std_val, title, xlabel, ylabel, ylim):
    fig1 = plt.figure(figsize=(10, 10))
    #plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(ylim)
    plt.errorbar(range(len(means_l)), means_l, stddevs_l, capsize=5, errorevery=2, label='loss')
    plt.errorbar(range(len(means_val)), means_val, std_val, capsize=5, errorevery=3, label='val_loss')
    plt.legend()
    filename = title +'.jpg'
    plt.savefig(filename)
    plt.close()

def make_figure_markers(means_mo, stddevs_mo, \
                        means_ma, stddevs_ma, \
                        means_w0, stddevs_w0, \
                        means_w1, stddevs_w1, \
                        means_w2, stddevs_w2, \
                        means_w3, stddevs_w3, \
                        means_w4, stddevs_w4, \
                        means_w5, stddevs_w5, \
                        title, xlabel, ylabel, ylim):
    fig1 = plt.figure(figsize=(10, 10))
    #plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(ylim)
    start, end = plt.get_xlim()
    plt.set_xticks(np.arange(start, end, 0.1))
    plt.errorbar(range(len(means_mo)), means_mo, stddevs_mo, capsize=5, errorevery=2, label='no attenuation')
    plt.errorbar(range(len(means_ma)), means_ma, stddevs_ma, capsize=5, errorevery=3, label='model')
    plt.errorbar(range(len(means_w0)), means_w0, stddevs_w0, capsize=5, errorevery=4, label='w0 set')
    plt.errorbar(range(len(means_w1)), means_w1, stddevs_w1, capsize=5, errorevery=5, label='w1 set')
    plt.errorbar(range(len(means_w2)), means_w2, stddevs_w2, capsize=5, errorevery=6, label='w2 set')
    plt.errorbar(range(len(means_w3)), means_w3, stddevs_w3, capsize=5, errorevery=7, label='w3 set')
    plt.errorbar(range(len(means_w4)), means_w4, stddevs_w4, capsize=5, errorevery=8, label='w4 set')
    plt.errorbar(range(len(means_w5)), means_w5, stddevs_w5, capsize=5, errorevery=9, label='w5 set')
    plt.legend(ncol=2, loc='lower right')
    filename = title + '.jpg'
    plt.savefig(filename)
    plt.close()

    _table_stats = []
    _table_p = []
    _line_s = []
    _line_p = []

    _line_s.append('MO')
    _line_p.append('MO')
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_mo, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_mo, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []

    _line_s.append('MA')
    _line_p.append('MA')
    stat, p = stats.ttest_ind(means_ma, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # ma ma
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_ma, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []

    _line_s.append('W0')
    _line_p.append('W0')
    stat, p = stats.ttest_ind(means_w0, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w0 w0
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w0, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W1')
    _line_p.append('W1')
    stat, p = stats.ttest_ind(means_w1, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w1 w1
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w1, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W2')
    _line_p.append('W2')
    stat, p = stats.ttest_ind(means_w2, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w2 w2
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w2, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W3')
    _line_p.append('W3')
    stat, p = stats.ttest_ind(means_w3, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w3 w3
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w3, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W4')
    _line_p.append('W4')
    stat, p = stats.ttest_ind(means_w4, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w4 w4
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w4, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W5')
    _line_p.append('W5')
    stat, p = stats.ttest_ind(means_w5, means_mo)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w5 w5
    _line_s.append('-')
    _line_p.append('-')
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    headers = ['', 'MO', 'MA', 'W0', 'W1', 'W2', 'W3', 'W4', 'W5']
    np.savetxt(title+'_stat.txt', ["%s" % tabulate.tabulate(_table_stats, headers, tablefmt="latex")], fmt='%s')
    np.savetxt(title+'_p.txt', ["%s" % tabulate.tabulate(_table_p, headers, tablefmt="latex")], fmt='%s')


def make_figure_iou(means_ma, stddevs_ma, \
                        means_w0, stddevs_w0, \
                        means_w1, stddevs_w1, \
                        means_w2, stddevs_w2, \
                        means_w3, stddevs_w3, \
                        means_w4, stddevs_w4, \
                        means_w5, stddevs_w5, \
                        title, xlabel, ylabel, ylim):
    fig1 = plt.figure(figsize=(10, 10))
    #plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(ylim)
    plt.errorbar(range(len(means_ma)), means_ma, stddevs_ma, capsize=5, errorevery=3, label='model')
    plt.errorbar(range(len(means_w0)), means_w0, stddevs_w0, capsize=5, errorevery=3, label='w0 set')
    plt.errorbar(range(len(means_w1)), means_w1, stddevs_w1, capsize=5, errorevery=3, label='w1 set')
    plt.errorbar(range(len(means_w2)), means_w2, stddevs_w2, capsize=5, errorevery=3, label='w2 set')
    plt.errorbar(range(len(means_w3)), means_w3, stddevs_w3, capsize=5, errorevery=3, label='w3 set')
    plt.errorbar(range(len(means_w4)), means_w4, stddevs_w4, capsize=5, errorevery=3, label='w4 set')
    plt.errorbar(range(len(means_w5)), means_w5, stddevs_w5, capsize=5, errorevery=3, label='w5 set')
    plt.legend(ncol=2, loc='lower right')
    filename = title + '.jpg'
    plt.savefig(filename)
    plt.close()

    _table_stats = []
    _table_p = []
    _line_s = []
    _line_p = []

    _line_s.append('MA')
    _line_p.append('MA')
    # ma ma
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_ma, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_ma, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []

    _line_s.append('W0')
    _line_p.append('W0')
    stat, p = stats.ttest_ind(means_w0, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w0 w0
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w0, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w0, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W1')
    _line_p.append('W1')
    stat, p = stats.ttest_ind(means_w1, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w1 w1
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w1, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w1, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W2')
    _line_p.append('W2')
    stat, p = stats.ttest_ind(means_w2, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w2 w2
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w2, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w2, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W3')
    _line_p.append('W3')
    stat, p = stats.ttest_ind(means_w3, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w3 w3
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w3, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w3, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W4')
    _line_p.append('W4')
    stat, p = stats.ttest_ind(means_w4, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w4, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w4 w4
    _line_s.append('-')
    _line_p.append('-')
    stat, p = stats.ttest_ind(means_w4, means_w5)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    _line_s.append('W5')
    _line_p.append('W5')
    stat, p = stats.ttest_ind(means_w5, means_ma)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w0)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w1)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w2)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w3)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    stat, p = stats.ttest_ind(means_w5, means_w4)
    _line_s.append(round(stat, 1))
    _line_p.append(round(p,4))
    # w5 w5
    _line_s.append('-')
    _line_p.append('-')
    _table_stats.append(_line_s)
    _table_p.append(_line_p)
    _line_s = []
    _line_p = []


    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    headers = ['',  'MA', 'W0', 'W1', 'W2', 'W3', 'W4', 'W5']
    np.savetxt(title+'_stat.txt', ["%s" % tabulate.tabulate(_table_stats, headers, tablefmt="latex")], fmt='%s')
    np.savetxt(title+'_p.txt', ["%s" % tabulate.tabulate(_table_p, headers, tablefmt="latex")], fmt='%s')


# run this from the exp_x folder
def do_stats_plot(num_runs,exp, do_iou):
    data_loss = []
    data_val_loss = []
    data_iou_main = []
    data_iou_custom_0 = []
    data_iou_custom_1 = []
    data_iou_custom_2 = []
    data_iou_custom_3 = []
    data_iou_custom_4 = []
    data_iou_custom_5 = []

    data_mkr_orig = []
    data_mkr_att = []
    data_mkr_att_custom_0 = []
    data_mkr_att_custom_1 = []
    data_mkr_att_custom_2 = []
    data_mkr_att_custom_3 = []
    data_mkr_att_custom_4 = []
    data_mkr_att_custom_5 = []
    # load results for each run of this experiment
    for run in range(num_runs):
        directory = 'run_' + str(run) + '/'
        data_loss.append(np.loadtxt(directory + 'plots/loss.txt'))
        data_val_loss.append(np.loadtxt(directory + 'plots/val_loss.txt'))

        if do_iou:
            data_iou_main.append(np.loadtxt(directory + 'plots/iou_main_model.txt'))

            data_iou_custom_0.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_0.txt'))
            data_iou_custom_1.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_1.txt'))
            data_iou_custom_2.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_2.txt'))
            data_iou_custom_3.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_3.txt'))
            data_iou_custom_4.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_4.txt'))
            data_iou_custom_5.append(np.loadtxt(directory + 'plots/iou_model_with_custom_weights_5.txt'))

        da_ = np.loadtxt(directory + 'plots/markers_in_original_img.txt')
        data_mkr_orig.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt')
        data_mkr_att.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt')
        data_mkr_att_custom_0.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt')
        data_mkr_att_custom_1.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt')
        data_mkr_att_custom_2.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt')
        data_mkr_att_custom_3.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt')
        data_mkr_att_custom_4.append(da_[start_marker_det_index:])
        da_ = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt')
        data_mkr_att_custom_5.append(da_[start_marker_det_index:])
    mean_loss = np.mean(np.asarray(data_loss), axis=0)
    mean_val_loss = np.mean(np.asarray(data_val_loss), axis=0)
    if do_iou:
        mean_iou_main = np.mean(np.asarray(data_iou_main), axis=0)
        mean_iou_custom_0 = np.mean(np.asarray(data_iou_custom_0), axis=0)
        mean_iou_custom_1 = np.mean(np.asarray(data_iou_custom_1), axis=0)
        mean_iou_custom_2 = np.mean(np.asarray(data_iou_custom_2), axis=0)
        mean_iou_custom_3 = np.mean(np.asarray(data_iou_custom_3), axis=0)
        mean_iou_custom_4 = np.mean(np.asarray(data_iou_custom_4), axis=0)
        mean_iou_custom_5 = np.mean(np.asarray(data_iou_custom_5), axis=0)

    mean_mkr_orig = np.mean(np.asarray(data_mkr_orig), axis=0)
    mean_mkr_att = np.mean(np.asarray(data_mkr_att), axis=0)
    mean_mkr_att_custom_0 = np.mean(np.asarray(data_mkr_att_custom_0), axis=0)
    mean_mkr_att_custom_1 = np.mean(np.asarray(data_mkr_att_custom_1), axis=0)
    mean_mkr_att_custom_2 = np.mean(np.asarray(data_mkr_att_custom_2), axis=0)
    mean_mkr_att_custom_3 = np.mean(np.asarray(data_mkr_att_custom_3), axis=0)
    mean_mkr_att_custom_4 = np.mean(np.asarray(data_mkr_att_custom_4), axis=0)
    mean_mkr_att_custom_5 = np.mean(np.asarray(data_mkr_att_custom_5), axis=0)

    stddev_loss = np.std(np.asarray(data_loss), axis=0)
    stddev_val_loss = np.std(np.asarray(data_val_loss), axis=0)

    if do_iou:
        stddev_iou_main = np.std(np.asarray(data_iou_main), axis=0)
        stddev_iou_custom_0 = np.std(np.asarray(data_iou_custom_0), axis=0)
        stddev_iou_custom_1 = np.std(np.asarray(data_iou_custom_1), axis=0)
        stddev_iou_custom_2 = np.std(np.asarray(data_iou_custom_2), axis=0)
        stddev_iou_custom_3 = np.std(np.asarray(data_iou_custom_3), axis=0)
        stddev_iou_custom_4 = np.std(np.asarray(data_iou_custom_4), axis=0)
        stddev_iou_custom_5 = np.std(np.asarray(data_iou_custom_5), axis=0)

    stddev_mkr_orig = np.std(np.asarray(data_mkr_orig), axis=0)
    stddev_mkr_att = np.std(np.asarray(data_mkr_att), axis=0)
    stddev_mkr_att_custom_0 = np.std(np.asarray(data_mkr_att_custom_0), axis=0)
    stddev_mkr_att_custom_1 = np.std(np.asarray(data_mkr_att_custom_1), axis=0)
    stddev_mkr_att_custom_2 = np.std(np.asarray(data_mkr_att_custom_2), axis=0)
    stddev_mkr_att_custom_3 = np.std(np.asarray(data_mkr_att_custom_3), axis=0)
    stddev_mkr_att_custom_4 = np.std(np.asarray(data_mkr_att_custom_4), axis=0)
    stddev_mkr_att_custom_5 = np.std(np.asarray(data_mkr_att_custom_5), axis=0)

    make_figure_loss(mean_loss, stddev_loss,mean_val_loss, stddev_val_loss, 'exp'+str(exp)+'_Mean_Loss', 'loss', 'epoch', [0.00003,0.0002])
    #make_figure(, 'Mean_Val_Loss', 'val_loss', 'epoch',[0,1])
    #make_figure(mean_iou, stddev_iou, 'exp'+str(exp)+'_Mean_Intersection_Over_Unit', 'IoU', 'epoch',[0,1])

    if do_iou:
        make_figure_iou(mean_iou_main, stddev_iou_main, \
                         mean_iou_custom_0, stddev_iou_custom_0, \
                         mean_iou_custom_1, stddev_iou_custom_1, \
                         mean_iou_custom_2, stddev_iou_custom_2, \
                         mean_iou_custom_3, stddev_iou_custom_3, \
                         mean_iou_custom_4, stddev_iou_custom_4, \
                         mean_iou_custom_5, stddev_iou_custom_5, \
                         'exp'+str(exp)+'_Mean_IoU', 'Intersection over Unit', 'epoch', [0,1])


    make_figure_markers(mean_mkr_orig, stddev_mkr_orig, \
                     mean_mkr_att, stddev_mkr_att, \
                     mean_mkr_att_custom_0, stddev_mkr_att_custom_0, \
                     mean_mkr_att_custom_1, stddev_mkr_att_custom_1, \
                     mean_mkr_att_custom_2, stddev_mkr_att_custom_2, \
                     mean_mkr_att_custom_3, stddev_mkr_att_custom_3, \
                     mean_mkr_att_custom_4, stddev_mkr_att_custom_4, \
                     mean_mkr_att_custom_5, stddev_mkr_att_custom_5, \
                     'exp'+str(exp)+'_Mean_Marker_Detection', 'Markers detected', 'epoch', [7,8.5])

def make_gif(folder, test_name, first_id, last_id, num_frames =20):
    images = []
    for i in range(num_frames):
        filename = folder+'pred_sequence_train_'+str(first_id)+ \
            '_' + str(last_id) + '_' + test_name + '_'+str(i)+'.png'
        images.append(imageio.imread(filename))
    imageio.mimsave(folder+'movie_'+str(first_id)+ \
            '_' + str(last_id)+ '_'+test_name+'.gif', images)

def do_ttest_self_other(num_runs):
    _self = []
    other = []
    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments'

    for run in range(num_runs):
        __self=np.loadtxt(multiple_experiments_folder+'/exp0/run_'+str(run)+'/plots/val_loss.txt')
        for i in range(10):
            _self.append(__self[i])
        _other=np.loadtxt(multiple_experiments_folder+'/exp3/run_'+str(run)+'/plots/val_loss.txt')
        for i in range(10):
            other.append(_other[i])
    _self = np.asarray(_self)
    other = np.asarray(other)
    stat, p = stats.ttest_ind(_self, other)
    #print('self ', _self)
    #print('other ', other)
    print('self mean ', np.mean(_self), ' std ', np.std(_self))
    print('other mean ', np.mean(other), ' std ', np.std(other))
    print('stat ', stat, ' p ', p)

def save_loss_and_val_loss(num_runs, exp_folder, exp_id, epochs=10):
    print('saving loss and val_loss')
    res_loss = []
    res_val_loss = []

    for run in range(num_runs):
        _res = np.loadtxt(exp_folder + '/run_'+str(run)+'/plots/loss.txt')
        _res_val = np.loadtxt(exp_folder + '/run_' + str(run) + '/plots/val_loss.txt')
        for i in range(epochs):
            res_loss.append(_res[i])
            res_val_loss.append(_res_val[i])

    loss_df = pd.DataFrame(np.asarray(res_loss), columns=['loss'])
    loss_df.to_csv('exp'+str(exp_id)+'_loss.csv')

    val_loss_df = pd.DataFrame(np.asarray(res_val_loss), columns=['val_loss'])
    val_loss_df.to_csv('exp'+str(exp_id)+'_val_loss.csv')
    print('saved')

def save_marker_detection(exp_id):
    print('saving marker detection results')
    p0_orig = []
    p0_att = []
    p0_att_w0 = []
    p0_att_w1 = []
    p0_att_w2 = []
    p0_att_w3 = []
    p0_att_w4 = []
    p0_att_w5 = []

    # load results for each run of this experiment
    for run in range(num_runs):
        directory = 'run_' + str(run) + '/'
        data = np.loadtxt(directory + 'plots/markers_in_original_img.txt')
        p0_orig.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt')
        p0_att.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt')
        p0_att_w0.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt')
        p0_att_w1.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt')
        p0_att_w2.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt')
        p0_att_w3.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt')
        p0_att_w4.append(data[start_marker_det_index:])
        data = np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt')
        p0_att_w5.append(data[start_marker_det_index:])


    # flatten list of lists
    p0_orig = [val for sublist in p0_orig for val in sublist]
    p0_att = [val for sublist in p0_att for val in sublist]
    p0_att_w0 = [val for sublist in p0_att_w0 for val in sublist]
    p0_att_w1 = [val for sublist in p0_att_w1 for val in sublist]
    p0_att_w2 = [val for sublist in p0_att_w2 for val in sublist]
    p0_att_w3 = [val for sublist in p0_att_w3 for val in sublist]
    p0_att_w4 = [val for sublist in p0_att_w4 for val in sublist]
    p0_att_w5 = [val for sublist in p0_att_w5 for val in sublist]

    p0_tuples = list(zip(p0_orig, p0_att, p0_att_w0, p0_att_w1, p0_att_w2, p0_att_w3, p0_att_w4, p0_att_w5))
    p0_df = pd.DataFrame(p0_tuples, columns=['orig','main', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5'])
    p0_df.to_csv('exp' + str(exp_id)+'_marker_detection_results.csv')

    print('saved')

if __name__ == "__main__":

    do_stats = True
    do_gif_videos = False
    do_self_other_test = False
    do_iou_plots = False

    starting_sample_for_gif = [125, 250, 375, 500, 625, 750, 875, 1000]
    num_frames = 20
    num_experiments = 3
    id_first_dyn_exp = 1 # id of the first dynamic experiment
    num_runs = 10

    if do_self_other_test:
        do_ttest_self_other(num_runs)
    plt.rcParams.update({'font.size': 18})

    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments'
    os.chdir(multiple_experiments_folder)
    for exp in range(num_experiments):
        exp = exp + id_first_dyn_exp
        exp_folder = multiple_experiments_folder + '/exp' + str(exp)
        os.chdir(exp_folder)

        print('doing plots for exp '+str(exp)+ ' run ')
        if do_stats:
            do_stats_plot(num_runs, exp, do_iou=do_iou_plots)
        if do_gif_videos:
            for run in range(num_runs):
                for i in range(len(starting_sample_for_gif)):
                    make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'attenuated', \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                    make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'imgp1', \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                    make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'predOF', \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                    make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'trueOF', \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                    make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'fw', \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                    for w in range(6):
                        make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'custom_fw_'+str(w), \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                        make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'custom_predOF_'+str(w), \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)
                        make_gif(exp_folder+'/run_'+str(run)+'/plots/gif/', 'attenuated_custom_'+str(w), \
                             starting_sample_for_gif[i], starting_sample_for_gif[i]+num_frames)

        save_marker_detection(exp)
        save_loss_and_val_loss(num_runs,exp_folder, exp)
        # go back
        os.chdir(multiple_experiments_folder)

    print('finished!')