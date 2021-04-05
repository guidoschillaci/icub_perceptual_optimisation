
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import os
from scipy import stats
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

        # flatten list of lists
        _data_loss = [val for sublist in _data_loss for val in sublist]
        _data_val_loss = [val for sublist in _data_val_loss for val in sublist]
        _data_iou = [val for sublist in _data_iou for val in sublist]
        _data_mkr_orig = [val for sublist in _data_mkr_orig for val in sublist]
        _data_mkr_att = [val for sublist in _data_mkr_att for val in sublist]
        _data_mkr_att_custom_0 = [val for sublist in _data_mkr_att_custom_0 for val in sublist]
        _data_mkr_att_custom_1 = [val for sublist in _data_mkr_att_custom_1 for val in sublist]
        _data_mkr_att_custom_2 = [val for sublist in _data_mkr_att_custom_2 for val in sublist]
        _data_mkr_att_custom_3 = [val for sublist in _data_mkr_att_custom_3 for val in sublist]
        _data_mkr_att_custom_4 = [val for sublist in _data_mkr_att_custom_4 for val in sublist]
        _data_mkr_att_custom_5 = [val for sublist in _data_mkr_att_custom_5 for val in sublist]

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

    print ('shape ', np.asarray(data_loss).shape)

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

    plots.make_figure_loss(mean_loss, stddev_loss,mean_val_loss, stddev_val_loss, 'exp'+str(exp)+'_Mean_Loss', 'loss', 'epoch', [0.00004,0.0006])
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
                     'exp'+str(exp)+'_Mean_Marker_Detection', 'Markers detected', 'epoch', [7.4,8.5])

def save_csv():
    p0_orig = []
    p0_att = []
    p0_att_w0 = []
    p0_att_w1 = []
    p0_att_w2 = []
    p0_att_w3 = []
    p0_att_w4 = []
    p0_att_w5 = []

    p1_orig = []
    p1_att = []
    p1_att_w0 = []
    p1_att_w1 = []
    p1_att_w2 = []
    p1_att_w3 = []
    p1_att_w4 = []
    p1_att_w5 = []

    p2_orig = []
    p2_att = []
    p2_att_w0 = []
    p2_att_w1 = []
    p2_att_w2 = []
    p2_att_w3 = []
    p2_att_w4 = []
    p2_att_w5 = []
    # load results for each run of this experiment
    for run in range(num_runs):
        directory = 'run_' + str(run) + '/phase_0/'
        p0_orig.append(np.loadtxt(directory + 'plots/markers_in_original_img.txt'))
        p0_att.append(np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt'))
        p0_att_w0.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt'))
        p0_att_w1.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt'))
        p0_att_w2.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt'))
        p0_att_w3.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt'))
        p0_att_w4.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt'))
        p0_att_w5.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt'))

        directory = 'run_' + str(run) + '/phase_1/'
        p1_orig.append(np.loadtxt(directory + 'plots/markers_in_original_img.txt'))
        p1_att.append(np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt'))
        p1_att_w0.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt'))
        p1_att_w1.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt'))
        p1_att_w2.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt'))
        p1_att_w3.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt'))
        p1_att_w4.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt'))
        p1_att_w5.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt'))

        directory = 'run_' + str(run) + '/phase_2/'
        p2_orig.append(np.loadtxt(directory + 'plots/markers_in_original_img.txt'))
        p2_att.append(np.loadtxt(directory + 'plots/markers_in_attenuated_img.txt'))
        p2_att_w0.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_0.txt'))
        p2_att_w1.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_1.txt'))
        p2_att_w2.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_2.txt'))
        p2_att_w3.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_3.txt'))
        p2_att_w4.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_4.txt'))
        p2_att_w5.append(np.loadtxt(directory + 'plots/markers_in_att_custom_weig_5.txt'))



    # flatten list of lists
    p0_orig = [val for sublist in p0_orig for val in sublist]
    p0_att = [val for sublist in p0_att for val in sublist]
    p0_att_w0 = [val for sublist in p0_att_w0 for val in sublist]
    p0_att_w1 = [val for sublist in p0_att_w1 for val in sublist]
    p0_att_w2 = [val for sublist in p0_att_w2 for val in sublist]
    p0_att_w3 = [val for sublist in p0_att_w3 for val in sublist]
    p0_att_w4 = [val for sublist in p0_att_w4 for val in sublist]
    p0_att_w5 = [val for sublist in p0_att_w5 for val in sublist]

    p1_orig = [val for sublist in p1_orig for val in sublist]
    p1_att = [val for sublist in p1_att for val in sublist]
    p1_att_w0 = [val for sublist in p1_att_w0 for val in sublist]
    p1_att_w1 = [val for sublist in p1_att_w1 for val in sublist]
    p1_att_w2 = [val for sublist in p1_att_w2 for val in sublist]
    p1_att_w3 = [val for sublist in p1_att_w3 for val in sublist]
    p1_att_w4 = [val for sublist in p1_att_w4 for val in sublist]
    p1_att_w5 = [val for sublist in p1_att_w5 for val in sublist]

    p2_orig = [val for sublist in p2_orig for val in sublist]
    p2_att = [val for sublist in p2_att for val in sublist]
    p2_att_w0 = [val for sublist in p2_att_w0 for val in sublist]
    p2_att_w1 = [val for sublist in p2_att_w1 for val in sublist]
    p2_att_w2 = [val for sublist in p2_att_w2 for val in sublist]
    p2_att_w3 = [val for sublist in p2_att_w3 for val in sublist]
    p2_att_w4 = [val for sublist in p2_att_w4 for val in sublist]
    p2_att_w5 = [val for sublist in p2_att_w5 for val in sublist]


    p0_tuples = list(zip(p0_orig, p0_att, p0_att_w0, p0_att_w1, p0_att_w2, p0_att_w3, p0_att_w4, p0_att_w5))
    p0_df = pd.DataFrame(p0_tuples, columns=['orig','main', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5'])
    p0_df.to_csv('raw_phase_0.csv')

    p1_tuples = list(zip(p1_orig, p1_att, p1_att_w0, p1_att_w1, p1_att_w2, p1_att_w3, p1_att_w4, p1_att_w5))
    p1_df = pd.DataFrame(p1_tuples, columns=['orig','main', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5'])
    p1_df.to_csv('raw_phase_1.csv')

    p2_tuples = list(zip(p2_orig, p2_att, p2_att_w0, p2_att_w1, p2_att_w2, p2_att_w3, p2_att_w4, p2_att_w5))
    p2_df = pd.DataFrame(p2_tuples, columns=['orig','main', 'w0', 'w1', 'w2', 'w3', 'w4', 'w5'])
    p2_df.to_csv('raw_phase_2.csv')


if __name__ == "__main__":
    do_stats = True

    plt.rcParams.update({'font.size': 18})
    num_experiments = 1
    id_first_dyn_exp = 6 # id of the first dynamic experiment
    num_runs = 10
    num_phases = 3
    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments'
    os.chdir(multiple_experiments_folder)
    for exp in range(num_experiments) + id_first_dyn_exp:
        exp_folder = multiple_experiments_folder + '/exp' + str(exp)
        os.chdir(exp_folder)
        for run in range(num_runs):
            print('doing plots for exp '+str(exp)+ ' run ' + str(run))
            if do_stats:
                do_stats_plot_incremental(num_runs, exp, num_phases)
                save_csv()

        # go back
        os.chdir(multiple_experiments_folder)