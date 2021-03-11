
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import os

import tkinter
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')


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
    plt.title(title)
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
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(ylim)
    plt.errorbar(range(len(means_mo)), means_mo, stddevs_mo, capsize=5, errorevery=2, label='original')
    plt.errorbar(range(len(means_ma)), means_ma, stddevs_ma, capsize=5, errorevery=3, label='attenuated')
    plt.errorbar(range(len(means_w0)), means_w0, stddevs_w0, capsize=5, errorevery=3, label='custom_w 0')
    plt.errorbar(range(len(means_w1)), means_w0, stddevs_w1, capsize=5, errorevery=3, label='custom_w 1')
    plt.errorbar(range(len(means_w2)), means_w0, stddevs_w2, capsize=5, errorevery=3, label='custom_w 2')
    plt.errorbar(range(len(means_w3)), means_w0, stddevs_w3, capsize=5, errorevery=3, label='custom_w 3')
    plt.errorbar(range(len(means_w4)), means_w0, stddevs_w4, capsize=5, errorevery=3, label='custom_w 4')
    plt.errorbar(range(len(means_w5)), means_w0, stddevs_w5, capsize=5, errorevery=3, label='custom_w 5')
    plt.legend()
    filename = title + '.jpg'
    plt.savefig(filename)
    plt.close()

# run this from the exp_x folder
def do_stats_plot(num_runs):
    data_loss = []
    data_val_loss = []
    data_iou = []
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
        data_iou.append(np.loadtxt(directory + 'plots/IoU.txt'))

        data_mkr_orig.append(np.loadtxt(directory + 'plots/markers_in_original_img.txt'))
        data_mkr_att.append(np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt'))
        data_mkr_att_custom_0.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt'))
        data_mkr_att_custom_1.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt'))
        data_mkr_att_custom_2.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt'))
        data_mkr_att_custom_3.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt'))
        data_mkr_att_custom_4.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt'))
        data_mkr_att_custom_5.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt'))
    mean_loss = np.mean(np.asarray(data_loss), axis=0)
    mean_val_loss = np.mean(np.asarray(data_val_loss), axis=0)
    mean_iou = np.mean(np.asarray(data_iou), axis=0)

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
    stddev_iou = np.std(np.asarray(data_iou), axis=0)

    stddev_mkr_orig = np.std(np.asarray(data_mkr_orig), axis=0)
    stddev_mkr_att = np.std(np.asarray(data_mkr_att), axis=0)
    stddev_mkr_att_custom_0 = np.std(np.asarray(data_mkr_att_custom_0), axis=0)
    stddev_mkr_att_custom_1 = np.std(np.asarray(data_mkr_att_custom_1), axis=0)
    stddev_mkr_att_custom_2 = np.std(np.asarray(data_mkr_att_custom_2), axis=0)
    stddev_mkr_att_custom_3 = np.std(np.asarray(data_mkr_att_custom_3), axis=0)
    stddev_mkr_att_custom_4 = np.std(np.asarray(data_mkr_att_custom_4), axis=0)
    stddev_mkr_att_custom_5 = np.std(np.asarray(data_mkr_att_custom_5), axis=0)

    make_figure_loss(mean_loss, stddev_loss,mean_val_loss, stddev_val_loss, 'Mean_Loss', 'loss', 'epoch', [0.00005,0.00022])
    #make_figure(, 'Mean_Val_Loss', 'val_loss', 'epoch',[0,1])
    make_figure(mean_iou, stddev_iou, 'Mean_Intersection_Over_Unit', 'IoU', 'epoch',[0,1])

    make_figure_loss(mean_mkr_orig, stddev_mkr_orig, \
                     mean_mkr_att, stddev_mkr_att, \
                     mean_mkr_att_custom_0, stddev_mkr_att_custom_0, \
                     mean_mkr_att_custom_1, stddev_mkr_att_custom_1, \
                     mean_mkr_att_custom_2, stddev_mkr_att_custom_2, \
                     mean_mkr_att_custom_3, stddev_mkr_att_custom_3, \
                     mean_mkr_att_custom_4, stddev_mkr_att_custom_4, \
                     mean_mkr_att_custom_5, stddev_mkr_att_custom_5, \
                     'Mean_Marker_Detection_in_Original_Img', 'Markers detected', 'epoch', [6,9])

if __name__ == "__main__":
    num_experiments = 4
    num_runs = 5
    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments_good'
    os.chdir(multiple_experiments_folder)
    for exp in range(num_experiments):
        exp_folder = multiple_experiments_folder + '/exp' + str(exp)
        os.chdir(exp_folder)
        do_stats_plot(num_runs)

        # go back
        os.chdir(multiple_experiments_folder)