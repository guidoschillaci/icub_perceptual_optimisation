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
            'directory_datasets': '',
            'directory_models': '',
            'model_filename':'model.h5',
            'model_plot_filename': 'model.png',

            'image_size': 64,
            'image_channels': 1,

            'test_dataset_factor': 0.1, # the size of the test dataset compared to the train
            'use_skin_data': False,

            'model_auxiliary': False, # use auxiliary weight model
            'model_batch_size': 32,
            'model_epochs': 50,
            'model_max_pool_size': 2,
            'model_conv_size': 3,

            'opt_flow_apply_threshold': False,  # if true, OF magnitudes below the following value are set to 0
            'opt_flow_treshold': 0.05, #was 0.02  # magnitudes below this value are set to 0. (original range btw 0 and 1)
            'opt_flow_only_magnitude': True,
            'opt_flow_max_value': 25,

            'make_plots': True,
            'plots_predict_size': 10,
            'verbosity_level': 3
        }

    def get(self, key_name):
        if key_name in self.dictionary.keys():
            return self.dictionary[key_name]
        else:
            print('Trying to access parameters key: '+ key_name+ ' which does not exist')

    def set(self, key_name, key_value):
        if key_name in self.dictionary.keys():
            print('Setting parameters key: ', key_name, ' to ', str(key_value))
            self.dictionary[key_name] = key_value
        else:
            print('Trying to modify parameters key: '+ key_name+ ' which does not exist')
            sys.exit(0)

    def save(self):
        # save as numpy array
        pickle.dump(self.dictionary, open(os.path.join(self.get('directory'), 'parameters.pkl'), 'wb'),  protocol=2) # protcolo2 for compatibility with python2
        # save also as plain text file
        with open(os.path.join(self.get('directory'), 'parameters.txt'), 'w') as f:
            print(self.dictionary, file=f)