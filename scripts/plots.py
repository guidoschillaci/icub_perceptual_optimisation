
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import os

def make_figure(means, stddevs, title, xlabel, ylabel):
    fig1 = plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.errorbar(range(len(means)), means, stddevs, capsize=5)#, errorevery=499)
    filename = title +'.jpg'
    plt.savefig(filename)
    plt.close()


# run this from the exp_x folder
def do_stats_plot(num_runs):
    data_loss = []
    data_val_loss = []
    data_iou = []
    # load results for each run of this experiment
    for run in range(num_runs):
        directory = 'run_' + str(run) + '/'
        data_loss.append(np.loadtxt(directory + 'plots/loss.txt'))
        data_val_loss.append(np.loadtxt(directory + 'plots/val_loss.txt'))
        data_iou.append(np.loadtxt(directory + 'plots/IoU.txt'))
    mean_loss = np.mean(np.asarray(data_loss), axis=0)
    mean_val_loss = np.mean(np.asarray(data_val_loss), axis=0)
    mean_iou = np.mean(np.asarray(data_iou), axis=0)

    stddev_loss = np.std(np.asarray(data_loss), axis=0)
    stddev_val_loss = np.std(np.asarray(data_val_loss), axis=0)
    stddev_iou = np.std(np.asarray(data_iou), axis=0)

    make_figure(mean_loss, stddev_loss, 'Mean_Loss', 'loss', 'epoch')
    make_figure(mean_val_loss, stddev_val_loss, 'Mean_Val_Loss', 'val_loss', 'epoch')
    make_figure(mean_iou, stddev_iou, 'Mean_Intersection_Over_Unit', 'IoU', 'epoch')


if __name__ == "__main__":
    num_experiments = 6
    num_runs = 5
    main_path = os.getcwd()
    multiple_experiments_folder = main_path + '/' + 'experiments'
    os.chdir(multiple_experiments_folder)
    for exp in range(num_experiments):
        exp_folder = multiple_experiments_folder + '/exp' + str(exp)
        os.chdir(exp_folder)
        do_stats_plot(num_runs)

        # go back
        os.chdir(multiple_experiments_folder)