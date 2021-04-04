
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
import plots


# run this from the exp_x folder
def do_stats_plot_incremental(num_runs,exp, num_phases=3):
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
        _data_loss = []
        _data_val_loss = []
        _data_iou = []
        _data_mkr_orig = []
        _data_mkr_att = []
        _data_mkr_att_custom_0 = []
        _data_mkr_att_custom_1 = []
        _data_mkr_att_custom_2 = []
        _data_mkr_att_custom_3 = []
        _data_mkr_att_custom_4 = []
        _data_mkr_att_custom_5 = []
        for phase in range(num_phases):
            directory = 'run_' + str(run) + '/phase_' + str(phase)+ '/'
            _data_loss.append(np.loadtxt(directory + 'plots/loss.txt'))
            _data_val_loss.append(np.loadtxt(directory + 'plots/val_loss.txt'))
            _data_iou.append(np.loadtxt(directory + 'plots/IoU.txt'))

            _data_mkr_orig.append(np.loadtxt(directory + 'plots/markers_in_original_img.txt'))
            _data_mkr_att.append(np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt'))
            _data_mkr_att_custom_0.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt'))
            _data_mkr_att_custom_1.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt'))
            _data_mkr_att_custom_2.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt'))
            _data_mkr_att_custom_3.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt'))
            _data_mkr_att_custom_4.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt'))
            _data_mkr_att_custom_5.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt'))
        data_loss.append(_data_loss)
        data_val_loss.append(_data_val_loss)
        data_iou.append(_data_iou)
        data_mkr_orig.append(_data_mkr_orig)
        data_mkr_att.append(_data_mkr_att)
        data_mkr_att_custom_0.append(_data_mkr_att_custom_0)
        data_mkr_att_custom_1.append(_data_mkr_att_custom_1)
        data_mkr_att_custom_2.append(_data_mkr_att_custom_2)
        data_mkr_att_custom_3.append(_data_mkr_att_custom_3)
        data_mkr_att_custom_4.append(_data_mkr_att_custom_4)
        data_mkr_att_custom_5.append(_data_mkr_att_custom_5)

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

    plots.make_figure_loss(mean_loss, stddev_loss,mean_val_loss, stddev_val_loss, 'exp'+str(exp)+'_Mean_Loss', 'loss', 'epoch', [0.00005,0.00022])
    #make_figure(, 'Mean_Val_Loss', 'val_loss', 'epoch',[0,1])
    plots.make_figure(mean_iou, stddev_iou, 'exp'+str(exp)+'_Mean_Intersection_Over_Unit', 'IoU', 'epoch',[0,1])

    plots.make_figure_markers(mean_mkr_orig, stddev_mkr_orig, \
                     mean_mkr_att, stddev_mkr_att, \
                     mean_mkr_att_custom_0, stddev_mkr_att_custom_0, \
                     mean_mkr_att_custom_1, stddev_mkr_att_custom_1, \
                     mean_mkr_att_custom_2, stddev_mkr_att_custom_2, \
                     mean_mkr_att_custom_3, stddev_mkr_att_custom_3, \
                     mean_mkr_att_custom_4, stddev_mkr_att_custom_4, \
                     mean_mkr_att_custom_5, stddev_mkr_att_custom_5, \
                     'exp'+str(exp)+'_Mean_Marker_Detection', 'Markers detected', 'epoch', [7,8.5])



if __name__ == "__main__":
    do_stats = True

    plt.rcParams.update({'font.size': 18})
    num_experiments = 1
    num_runs = 10
    num_phases = 3
    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments_paper'
    os.chdir(multiple_experiments_folder)
    for exp in range(num_experiments):
        exp_folder = multiple_experiments_folder + '/exp' + str(exp)
        os.chdir(exp_folder)
        for run in range(num_runs):
            print('doing plots for exp '+str(exp)+ ' run ' + str(run))
            if do_stats:
                do_stats_plot_incremental(num_runs, exp, num_phases)

        # go back
        os.chdir(multiple_experiments_folder)