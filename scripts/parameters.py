from __future__ import print_function # added this to support usage of python2

import os
import numpy as np
from enum import Enum
import pickle
import sys

class Parameters:

    def __init__(self):
        self.dictionary = {
            'directory_results': '',
            'directory_plots': '',
            'directory_plots_gif': '',
            'directory_datasets_train': '',
            'directory_datasets_test': '',
            'directory_datasets': '',
            'directory_models': '',
            'model_filename':'model.h5',
            'model_plot_filename': 'model.png',
            'save_sequence_plots': True,
            'save_sequence_plots_gif': False, # all the gif files

            'image_size': 32, # size of the input image in the model
            'image_original_shape': None, # shape of the image in the original dataset
            'image_channels': 1,

            'dataset_train_type': 'icub_and_ball', # 'icub_alone' or 'icub_and_ball' or 'only_ball'
            'dataset_test_type': 'icub_and_ball',  # 'icub_alone' or 'icub_and_ball' or 'only_ball'
            'dataset_test_shuffle': True,
            #'dataset_shuffle_seed': 42,
            'test_dataset_factor': 0.05, # the portion of the train dataset to be used as test
            'dataset_split_seed': 33, # the portion of the train dataset to be used as test
            'use_skin_data': False,
            'filter_out_background_images': False,
            'threshold_for_background_img': 80, #if moving pixels are less than this number, don't include the sample in the dataset

            #'model_custom_training_loop': False, # if False, use standard keras fit functions
            'model_auxiliary': True, # use auxiliary weight model
            'model_batch_size': 32,
            'model_epochs': 10,
            'model_max_pool_size': 2,
            'model_conv_size': 3,
            'model_sensor_fusion_alpha': 0.5,
            'model_sensor_fusion_beta': 0.05, # was 0.05
            'model_num_modalities': 3, # visual, proprioceptive, motor
            'model_use_activity_regularization_layer': True,

            'opt_flow_apply_threshold': False,  # if true, OF magnitudes below the following value are set to 0
            'opt_flow_binarize': True,  # if true, opt_flow (only for sensory attenuation) will have only 0 or 1 values, according to the threhsold specified below
            #'opt_flow_treshold': 0.02, #was 0.02  # magnitudes below this value are set to 0. (original range btw 0 and 1)
            'opt_flow_max_value': 52,
            'opt_flow_binary_threshold': 0.001, # below this, is set to 0, higher than this, it is set to 1

            'make_plots': True,
            'plots_predict_size': 10,
            'verbosity_level': 1
        }

    def get(self, key_name):
        if key_name in self.dictionary.keys():
            return self.dictionary[key_name]
        else:
            print('Trying to access parameters key: '+ key_name+ ' which does not exist')
            sys.exit(0)

    def set(self, key_name, key_value):
        if key_name in self.dictionary.keys():
            print('Setting parameters key: ', key_name, ' to ', str(key_value))
            self.dictionary[key_name] = key_value
        else:
            print('Trying to modify parameters key: '+ key_name+ ' which does not exist')
            sys.exit(0)

    def save(self):
        # save as numpy array
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory_results'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory_results'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)

    def print(self):
        print('Parameters:')
        for keys, values in self.dictionary.items():
            print(keys, ': ', values)