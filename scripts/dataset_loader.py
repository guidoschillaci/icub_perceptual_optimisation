import numpy as np
import os
from sklearn import preprocessing
import random
from copy import deepcopy
import cv2
import time
from tqdm import tqdm
import sys

class DatasetLoader():

    def __init__(self, param):
        self.parameters = param
        #self.load_dataset()

    def split_train_test(self):
        print('splitting train/test datasets')
        self.len_original_dataset = len(self.dataset_joints)
        self.len_test_dataset = int(self.len_original_dataset* self.parameters.get('test_dataset_factor'))
        # set fixed seed to get always the same test indexes (for comparison between runs)
        np.random.seed(42)
        self.test_indexes = random.sample(list(np.arange(self.len_original_dataset)), self.len_test_dataset)
        # reset the seed
        np.random.seed(int(time.time()))
        self.train_indexes = np.ones(self.len_original_dataset, np.bool)
        self.train_indexes[self.test_indexes] = 0


        #print('len original ', self.len_original_dataset)
        #print('len test ', len(test_indexes))
        #print('len train', len(train_indexes))
        '''
        # train datasets
        self.train_ds_images_t = (self.dataset_images_t[train_indexes])
        self.train_ds_images_tp1 = (self.dataset_images_tp1[train_indexes])
        self.train_ds_opt_flow = (self.dataset_optical_flow[train_indexes])
        self.train_ds_joints = (self.dataset_joints[train_indexes])
        self.train_ds_cmd = (self.dataset_cmd[train_indexes])
        if self.parameters.get('use_skin_data'):
            self.train_ds_skin = (self.dataset_skin_values[train_indexes])
        # test datasets
        self.test_ds_images_t = (self.dataset_images_t[test_indexes])
        self.test_ds_images_tp1 = (self.dataset_images_tp1[test_indexes])
        self.test_ds_opt_flow = (self.dataset_optical_flow[test_indexes])
        self.test_ds_joints = (self.dataset_joints[test_indexes])
        self.test_ds_cmd = (self.dataset_cmd[test_indexes])
        if self.parameters.get('use_skin_data'):
            self.test_ds_skin = (self.dataset_skin_values[test_indexes])

        '''

    def load_datasets(self):
        print('loading datasets')

        # images at time t
        if self.parameters.get('image_channels')==1:
            self.dataset_images_t_orig = np.load(self.parameters.get('directory_datasets')+'dataset_images_grayscale.npy')
            if self.parameters.get('image_size') != 64:
                self.dataset_images_t = []
                for i in tqdm(range(len(self.dataset_images_t_orig))):
                    cv2_img = cv2.resize(self.dataset_images_t_orig[i], (self.parameters.get('image_size'), self.parameters.get('image_size')), interpolation=cv2.INTER_LINEAR)
                    self.dataset_images_t.append( np.array(cv2_img))
            else:
                self.dataset_images_t = self.dataset_images_t_orig
        else:
            self.dataset_images_t_orig = np.load(self.parameters.get('directory_datasets') + 'dataset_images.npy')
            if self.parameters.get('image_size') != 64:
                self.dataset_images_t = []
                for i in tqdm(range(len(self.dataset_images_t_orig))):
                    cv2_img = cv2.resize(self.dataset_images_t_orig[i], (self.parameters.get('image_size'), self.parameters.get('image_size'), self.parameters.get('image_channels')), interpolation=cv2.INTER_LINEAR)
                    self.dataset_images_t.append( np.array(cv2_img))
            else:
                self.dataset_images_t = self.dataset_images_t_orig



        # image at time t+1
        #if self.parameters.get('image_channels') == 1:
        #    self.dataset_images_tp1 = np.load(self.parameters.get('directory_datasets') + 'dataset_images_grayscale.npy')
        #else:
        #    self.dataset_images_tp1 = np.load(self.parameters.get('directory_datasets') + 'dataset_images.npy')
        self.dataset_images_tp1 = deepcopy(self.dataset_images_t)

        # starts from t+1
        self.dataset_images_tp1 = np.delete(self.dataset_images_tp1, 0, 0)
        # pop last elements from the datasets to match the size of self.dataset_images_tp1
        self.dataset_images_t = np.delete(self.dataset_images_t, len(self.dataset_images_t) - 1, 0)

        self.dataset_joints = np.load(self.parameters.get('directory_datasets')+'dataset_joint_encoders.npy')
        self.dataset_cmd = np.load(self.parameters.get('directory_datasets')+'dataset_motor_commands.npy')
        if self.parameters.get('use_skin_data'):
            self.dataset_skin_values = np.load(self.parameters.get('directory_datasets')+'dataset_skin_values.npy')
        self.dataset_timestamps = np.load(self.parameters.get('directory_datasets')+'dataset_timestamps.npy')

        # pop last elements from the datasets to match the size of self.dataset_images_tp1
        self.dataset_joints = np.delete(self.dataset_joints, len(self.dataset_joints)-1, 0)
        self.dataset_cmd = np.delete(self.dataset_cmd, len(self.dataset_cmd)-1, 0)
        if self.parameters.get('use_skin_data'):
            self.dataset_skin_values = np.delete(self.dataset_skin_values, len(self.dataset_skin_values)-1, 0)
        self.dataset_timestamps = np.delete(self.dataset_timestamps, len(self.dataset_timestamps)-1, 0)

        if os.path.isfile(self.parameters.get('directory_datasets') + 'dataset_optical_flow.npy'):
            print('loading optical flow dataset')
            self.dataset_optical_flow = np.load(self.parameters.get('directory_datasets') + 'dataset_optical_flow.npy')
        else:
            print('Dataset optical flow does not exists. Creating it now...')
            self.dataset_optical_flow = []
            for i in tqdm(range(len(self.dataset_images_t))):
                if self.parameters.get('opt_flow_only_magnitude'):
                    opt_flow_polar_normalised = np.zeros((self.parameters.get('image_size'), self.parameters.get('image_size'),1))
                else:
                    opt_flow_polar_normalised = np.zeros((self.parameters.get('image_size'), self.parameters.get('image_size'), 3))
                flow = cv2.calcOpticalFlowFarneback(self.dataset_images_t[i], self.dataset_images_tp1[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                opt_flow_polar_normalised[..., 0] = (magnitude) # cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
                if not self.parameters.get('opt_flow_only_magnitude'):
                    opt_flow_polar_normalised[..., 1] = np.cos(ang) # ensures this is within -1,1 and doesnt' jump from 0to 180
                    opt_flow_polar_normalised[..., 2] = np.sin(ang) # ensures this is within -1,1 and doesnt' jump from 0to 180
                self.dataset_optical_flow.append(opt_flow_polar_normalised)
            self.dataset_optical_flow = np.array(self.dataset_optical_flow)

            if self.parameters.get('opt_flow_only_magnitude'):
                max_optflow = np.max(np.asarray(self.dataset_optical_flow).flatten())
                self.dataset_optical_flow = self.dataset_optical_flow/max_optflow
                if self.parameters.get('opt_flow_apply_threshold'):
                    magnitude[ magnitude < self.parameters.get('opt_flow_treshold')] = 0
                self.parameters.set('opt_flow_max_value', max_optflow)
            else:
                print('DS loader. TODO. normalisation for Optical flow with angle data has not been implemented yet.')
                sys.exit(0)

            print('saving optical flow dataset')
            np.save(self.parameters.get('directory_datasets')+'dataset_optical_flow.npy', self.dataset_optical_flow)

        #self.scaler_dataset_img = preprocessing.MinMaxScaler()
        #self.scaler_dataset_opt_flow = preprocessing.MinMaxScaler()
        self.scaler_dataset_joints = preprocessing.MinMaxScaler()
        self.scaler_dataset_cmd = preprocessing.MinMaxScaler()
        if self.parameters.get('use_skin_data'):
            self.scaler_dataset_skin = preprocessing.MinMaxScaler()

        #normalise images
        self.dataset_images_t = self.dataset_images_t / 255.
        self.dataset_images_tp1 = self.dataset_images_tp1 / 255.
        # normalise optical flow
        #max_optflow = np.max(np.asarray(self.dataset_optical_flow).flatten())
        #self.dataset_optical_flow = self.dataset_optical_flow/max_optflow


        #if self.parameters.get('verbosity_level') >= 2:
        #    print('min opt', np.min(np.asarray(self.dataset_optical_flow).flatten()))
        #    print('max opt', max_optflow)
        #    print('max opt norm', np.max(np.asarray(self.dataset_optical_flow).flatten())/max_optflow)

        self.dataset_joints = self.scaler_dataset_joints.fit_transform(self.dataset_joints)
        self.dataset_cmd = self.scaler_dataset_cmd.fit_transform(self.dataset_cmd)
        if self.parameters.get('use_skin_data'):
            self.dataset_skin_values = self.scaler_dataset_skin.fit_transform(self.dataset_skin_values)

        if self.parameters.get('verbosity_level') >= 2:
            print('nr images (t): ', str(np.asarray(self.dataset_images_t).shape))
            print('img(0) shape: ', str(np.asarray(self.dataset_images_t[0]).shape))

            print('nr images (t+1): ', str(np.asarray(self.dataset_images_tp1).shape))
            print('img_tp1(0) shape: ', str(np.asarray(self.dataset_images_tp1[0]).shape))

            print('nr optflow: ', str(np.asarray(self.dataset_optical_flow).shape))
            print('optflow(0) shape: ', str(np.asarray(self.dataset_optical_flow[0]).shape))

            print('nr joints: ', str(np.asarray(self.dataset_joints).shape))
            print('joints(0) shape: ', str(np.asarray(self.dataset_joints[0]).shape))

            print('nr cmd: ', str(np.asarray(self.dataset_cmd).shape))
            print('cmd(0) shape: ', str(np.asarray(self.dataset_cmd[0]).shape))

            if self.parameters.get('use_skin_data'):
                print('nr skin_values: ', str(np.asarray(self.dataset_skin_values).shape))
                print('skin_values(0) shape: ', str(np.asarray(self.dataset_skin_values[0]).shape))

            print('nr timestamps: ', str(np.asarray(self.dataset_timestamps).shape))
            print('timestamps(0) shape: ', str(np.asarray(self.dataset_timestamps[0]).shape))


        # reshape needed for model training
        self.dataset_images_t = self.dataset_images_t.reshape(len(self.dataset_images_t), \
                                                              self.parameters.get('image_size'), \
                                                              self.parameters.get('image_size'), \
                                                              self.parameters.get('image_channels'))
        self.dataset_images_tp1 = self.dataset_images_tp1.reshape(len(self.dataset_images_tp1), \
                                                              self.parameters.get('image_size'), \
                                                              self.parameters.get('image_size'), \
                                                              self.parameters.get('image_channels'))

        if self.parameters.get('opt_flow_only_magnitude'):
            self.dataset_optical_flow = self.dataset_optical_flow.reshape(len(self.dataset_optical_flow), \
                                                                          self.parameters.get('image_size'), \
                                                                          self.parameters.get('image_size'), \
                                                                          1)
        else:
            self.dataset_optical_flow = self.dataset_optical_flow.reshape(len(self.dataset_optical_flow), \
                                                                          self.parameters.get('image_size'), \
                                                                          self.parameters.get('image_size'), \
                                                                          3)


        self.split_train_test()
        print ('Datasets loaded!')