import numpy as np
import os
from sklearn import preprocessing
import random
from copy import deepcopy
import cv2
import time
from tqdm import tqdm
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import utils

class Dataset():
    def __init__(self, param):
        self.parameters = param
        # type of dataset, should be set to 'train' or 'test'
        self.type = None
        # indexes of the train and test elements
        #self.idx_train = []
        #self.idx_test = []
        # the images read from the file
        self.images_raw = []
        # the pre-processed images datasets, aligned at time t (its size is len(self.dataset_images_raw)-1)
        # with the same shape as the raw images
        self.images_orig_size_t = []
        # with the shape needed for the forward model
        self.images_t = []
        # the pre-processed images datasets, aligned at time t+1 (its size is len(self.dataset_images_raw)-1)
        # with the same shape as the raw images
        self.images_orig_size_tp1 = []
        # with the shape needed for the forward model
        self.images_tp1 = []
        # the joints position at time t
        self.joints = []
        # the motor commands applied at time t
        self.cmd = []
        # the artificial skin activations at time t
        if self.parameters.get('use_skin_data'):
            self.skin_values = []
        # the timestamps at time t
        self.timestamps = []
        # the optical flow between images at t and t+1
        self.optical_flow = []


class DatasetLoader():

    def __init__(self, param):
        self.scaler_already_set = False
        self.parameters = param
        self.train = Dataset(param)
        self.test = Dataset(param)
        #self.train_unshuffled = Dataset(param)

    def split_train_test(self, dataset, dataset_type):
        print('splitting train/test datasets')
        if dataset_type == 'train':
            dataset.images_t, _ = train_test_split(dataset.images_t,
                                                   test_size = self.parameters.get('test_dataset_factor'),
                                                   random_state = self.parameters.get('dataset_split_seed'))
            dataset.images_orig_size_t, _ = train_test_split(dataset.images_orig_size_t,
                                                             test_size=self.parameters.get('test_dataset_factor'),
                                                             random_state=self.parameters.get('dataset_split_seed'))
            dataset.images_tp1, _ = train_test_split(dataset.images_tp1,
                                                     test_size=self.parameters.get('test_dataset_factor'),
                                                     random_state=self.parameters.get('dataset_split_seed'))
            dataset.images_orig_size_tp1, _ = train_test_split(dataset.images_orig_size_tp1,
                                                               test_size=self.parameters.get('test_dataset_factor'),
                                                               random_state=self.parameters.get('dataset_split_seed'))
            dataset.joints, _ = train_test_split(dataset.joints,
                                                 test_size = self.parameters.get('test_dataset_factor'),
                                                 random_state = self.parameters.get('dataset_split_seed'))
            dataset.cmd, _ = train_test_split(dataset.cmd,
                                              test_size = self.parameters.get('test_dataset_factor'),
                                              random_state = self.parameters.get('dataset_split_seed'))
            dataset.optical_flow, _ = train_test_split(dataset.optical_flow,
                                                       test_size = self.parameters.get('test_dataset_factor'),
                                                       random_state = self.parameters.get('dataset_split_seed'))
        else: # test
            if self.parameters.get('dataset_test_shuffle'):
                _, dataset.images_t = train_test_split(dataset.images_t,
                                                       test_size = self.parameters.get('test_dataset_factor'),
                                                       random_state = self.parameters.get('dataset_split_seed'))
                _, dataset.images_orig_size_t = train_test_split(dataset.images_orig_size_t,
                                                                 test_size=self.parameters.get('test_dataset_factor'),
                                                                 random_state=self.parameters.get('dataset_split_seed'))
                _, dataset.images_tp1 = train_test_split(dataset.images_tp1,
                                                         test_size=self.parameters.get('test_dataset_factor'),
                                                         random_state=self.parameters.get('dataset_split_seed'))
                _, dataset.images_orig_size_tp1 = train_test_split(dataset.images_orig_size_tp1,
                                                                   test_size=self.parameters.get('test_dataset_factor'),
                                                                   random_state=self.parameters.get('dataset_split_seed'))
                _, dataset.joints = train_test_split(dataset.joints,
                                                     test_size = self.parameters.get('test_dataset_factor'),
                                                     random_state = self.parameters.get('dataset_split_seed'))
                _, dataset.cmd = train_test_split(dataset.cmd,
                                                  test_size = self.parameters.get('test_dataset_factor'),
                                                  random_state = self.parameters.get('dataset_split_seed'))
                _, dataset.optical_flow = train_test_split(dataset.optical_flow,
                                                           test_size = self.parameters.get('test_dataset_factor'),
                                                           random_state = self.parameters.get('dataset_split_seed'))
            else: # unshuffled test dataset, use this only for plotting sequences
                index_from = len(dataset.images_t) - int(len(dataset.images_t) * self.parameters.get('test_dataset_factor')) -1
                dataset.images_t = deepcopy(dataset.images_t[index_from:])
                dataset.images_orig_size_t = deepcopy(dataset.images_orig_size_t[index_from:])
                dataset.images_tp1 = deepcopy(dataset.images_tp1[index_from:])
                dataset.images_orig_size_tp1 = deepcopy(dataset.images_orig_size_tp1[index_from:])
                dataset.joints = deepcopy(dataset.joints[index_from:])
                dataset.cmd = deepcopy(dataset.cmd[index_from:])
                dataset.optical_flow = deepcopy(dataset.optical_flow[index_from:])

    #def binarize_optical_flow(self, optflow, positive_value = 1):
    #    if self.parameters.get('opt_flow_binarize'):
    #        return np.array(np.where(optflow > self.parameters.get('opt_flow_binary_threshold'), positive_value, 0), dtype='uint8')
    #    return np.array(optflow, dtype='uint8')

    def filter_out_samples_with_non_moving_objects(self,dataset):
        print('filtering out samples with non-moving objects')
        indexes = []
        for i in tqdm(range(len(dataset.optical_flow))):
            binarised_of = utils.binarize_optical_flow(self.parameters, dataset.optical_flow[i])
            if np.count_nonzero(binarised_of == 1) < self.parameters.get('threshold_for_background_img'):
                # nothing is moving in the visual input. remove this sample
                indexes.append(i)
                #if random.random() < 0.01:
                #    cv2.imwrite(self.parameters.get('directory_plots')+'image_removed_'+str(i)+'.png', dataset.images_t[i]*255)
        print('total to be removed = ', str(len(indexes)))
        dataset.images_t = np.delete(dataset.images_t, indexes,0)
        dataset.images_orig_size_t = np.delete(dataset.images_orig_size_t, indexes,0)
        dataset.images_tp1 = np.delete(dataset.images_tp1, indexes,0)
        dataset.images_orig_size_tp1 = np.delete(dataset.images_orig_size_tp1, indexes,0)
        dataset.joints =  np.delete(dataset.joints, indexes,0)
        dataset.cmd =  np.delete(dataset.cmd, indexes,0)
        dataset.optical_flow =  np.delete(dataset.optical_flow, indexes,0)
        print('new dataset lenght:', str(len(dataset.optical_flow)))

    def load_datasets(self):
        print('loading datasets')
        self.load_datasets_from_folder(self.train, self.parameters.get('directory_datasets_train'), type = 'train')
        self.load_datasets_from_folder(self.test, self.parameters.get('directory_datasets_test'), type = 'test')

    def load_datasets_from_folder(self, dataset, folder, type):
        if type == 'train':
            print('train ds')
        else:
            print('test ds')

        # images at time t
        if self.parameters.get('image_channels')==1:
            dataset.images_raw = np.load(folder+'dataset_images_grayscale.npy')
            #dataset.images_raw = np.load(folder + 'dataset_images.npy')
            #print ('dataset image raw size', dataset.images_raw[0].shape)
            #cv2.imwrite('image_raw.png', dataset.images_raw[0])
            # the background image to be used in the sensory attenuation process
            dataset.background_image = cv2.imread(self.parameters.get('directory_datasets')+'background_image.png', cv2.IMREAD_GRAYSCALE)
            if self.parameters.get('image_original_shape') is None:
                self.parameters.set('image_original_shape', (dataset.background_image.shape[1],\
                                                             dataset.background_image.shape[0]))

            for i in tqdm(range(len(dataset.images_raw)-1)):
                #print('dataset image raw max ', np.max(dataset.images_raw[i]), ' min ', np.min(dataset.images_raw[0]))
                cv2_img = cv2.resize(dataset.images_raw[i], (self.parameters.get('image_size'), self.parameters.get('image_size')))
                dataset.images_t.append( np.array(cv2_img))
                dataset.images_orig_size_t.append(dataset.images_raw[i])
            print('dataset image images_orig_size_t size', dataset.images_orig_size_t[0].shape)
        else:
            dataset.images_raw = np.load(folder + 'dataset_images.npy')
            dataset.background_image = cv2.imread(folder + 'background_image.png')
            for i in tqdm(range(len(dataset.images_raw)-1)):
                cv2_img_reshaped = cv2.resize(dataset.images_raw[i], (self.parameters.get('image_size'), self.parameters.get('image_size'), self.parameters.get('image_channels')))
                dataset.images_t.append( np.array(cv2_img_reshaped))
                dataset.images_orig_size_t.append(dataset.images_raw[i])
                if self.parameters.get('image_original_shape') is None:
                    self.parameters.set('image_original_shape', dataset.background_image.shape)

        for i in tqdm(range(len(dataset.images_raw)-1)):
            cv2_img_reshaped = cv2.resize(dataset.images_raw[i+1],
                                 (self.parameters.get('image_size'), self.parameters.get('image_size')),
                                 interpolation=cv2.INTER_LINEAR)
            dataset.images_tp1.append(np.array(cv2_img_reshaped))
            dataset.images_orig_size_tp1.append(dataset.images_raw[i+1])

        dataset.joints = np.load(folder+'dataset_joint_encoders.npy')
        dataset.cmd = np.load(folder+'dataset_motor_commands.npy')
        if self.parameters.get('use_skin_data'):
            dataset.skin_values = np.load(folder+'dataset_skin_values.npy')
        dataset.timestamps = np.load(folder+'dataset_timestamps.npy')

        # pop last elements from the datasets to match the size of self.dataset_images_tp1
        dataset.joints = np.delete(dataset.joints, len(dataset.joints)-1, 0)
        dataset.cmd = np.delete(dataset.cmd, len(dataset.cmd)-1, 0)
        if self.parameters.get('use_skin_data'):
            dataset.skin_values = np.delete(dataset.skin_values, len(dataset.skin_values)-1, 0)
        dataset.timestamps = np.delete(dataset.timestamps, len(dataset.timestamps)-1, 0)

        if os.path.isfile(folder + 'dataset_optical_flow.npy'):
            print('loading optical flow dataset')
            dataset.optical_flow = np.load(folder + 'dataset_optical_flow.npy')
        else:
            print('Dataset optical flow does not exists. Creating it now...')
            for i in tqdm(range(len(dataset.images_t))):
                opt_flow_polar_normalised = np.zeros((self.parameters.get('image_size'), self.parameters.get('image_size'),1))
                flow = cv2.calcOpticalFlowFarneback(dataset.images_t[i], dataset.images_tp1[i], None, 0.5, 3, 5, 3, 5, 0.9, 0)
                magnitude, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                dataset.optical_flow.append(utils.binarize_optical_flow(self.parameters,magnitude,positive_value=1.0))
            dataset.optical_flow = np.array(dataset.optical_flow)
            print ('dataset max ', np.max(np.asarray(dataset.optical_flow).flatten()), \
                   ' min ',  np.min(np.asarray(dataset.optical_flow).flatten()), \
                   ' mean ',  np.mean(np.asarray(dataset.optical_flow).flatten()))

            max_optflow = np.max(np.asarray(dataset.optical_flow).flatten())
            #dataset.optical_flow = dataset.optical_flow/self.parameters.get('opt_flow_max_value')
            #max_optflow

            print('saving optical flow dataset')
            np.save(folder+'dataset_optical_flow.npy', dataset.optical_flow)

        if not self.scaler_already_set:
            self.scaler_dataset_joints = preprocessing.MinMaxScaler()
            self.scaler_dataset_cmd = preprocessing.MinMaxScaler()
            self.scaler_dataset_joints.fit(dataset.joints)
            self.scaler_dataset_cmd.fit(dataset.cmd)
            if self.parameters.get('use_skin_data'):
                self.scaler_dataset_skin = preprocessing.MinMaxScaler()
                self.scaler_dataset_skin.fit(dataset.skin_values)

        #normalise images
        dataset.images_t = np.asarray(dataset.images_t) / 255.
        dataset.images_tp1 = np.asarray(dataset.images_tp1) / 255.
        dataset.joints = self.scaler_dataset_joints.transform(dataset.joints)
        dataset.cmd = self.scaler_dataset_cmd.transform(dataset.cmd)
        if self.parameters.get('use_skin_data'):
            dataset.skin_values = self.scaler_dataset_skin.transform(dataset.skin_values)

        if self.parameters.get('verbosity_level') >= 2:
            print('nr images (t): ', str(np.asarray(dataset.images_t).shape))
            print('img(0) shape: ', str(np.asarray(dataset.images_t[0]).shape))

            print('nr images (t+1): ', str(np.asarray(dataset.images_tp1).shape))
            print('img_tp1(0) shape: ', str(np.asarray(dataset.images_tp1[0]).shape))

            print('nr optflow: ', str(np.asarray(dataset.optical_flow).shape))
            print('optflow(0) shape: ', str(np.asarray(dataset.optical_flow[0]).shape))

            print('nr joints: ', str(np.asarray(dataset.joints).shape))
            print('joints(0) shape: ', str(np.asarray(dataset.joints[0]).shape))

            print('nr cmd: ', str(np.asarray(dataset.cmd).shape))
            print('cmd(0) shape: ', str(np.asarray(dataset.cmd[0]).shape))

            if self.parameters.get('use_skin_data'):
                print('nr skin_values: ', str(np.asarray(dataset.skin_values).shape))
                print('skin_values(0) shape: ', str(np.asarray(dataset.skin_values[0]).shape))

            print('nr timestamps: ', str(np.asarray(dataset.timestamps).shape))
            print('timestamps(0) shape: ', str(np.asarray(dataset.timestamps[0]).shape))

        # reshape needed for model training
        dataset.images_t = dataset.images_t.reshape(len(dataset.images_t), \
                                                    self.parameters.get('image_size'), \
                                                    self.parameters.get('image_size'), \
                                                    self.parameters.get('image_channels'))
        dataset.images_tp1 = dataset.images_tp1.reshape(len(dataset.images_tp1), \
                                                        self.parameters.get('image_size'), \
                                                        self.parameters.get('image_size'), \
                                                        self.parameters.get('image_channels'))

        #if self.parameters.get('opt_flow_only_magnitude'):
        dataset.optical_flow = dataset.optical_flow.reshape(len(dataset.optical_flow), \
                                                            self.parameters.get('image_size'), \
                                                            self.parameters.get('image_size'), \
                                                            1)
        #else:
        #    dataset.optical_flow = dataset.optical_flow.reshape(len(dataset.optical_flow), \
        #                                                        self.parameters.get('image_size'), \
        #                                                        self.parameters.get('image_size'), \
        #                                                        3)
        #if type =='train':
        #    #self.train_unshuffled = deepcopy(self.train)
        #    self.train_unshuffled = self.train

        if self.parameters.get('filter_out_background_images'):
            self.filter_out_samples_with_non_moving_objects(dataset)

        self.split_train_test(dataset, type)
        # free memory
        del dataset.images_raw
        print ('Datasets loaded!')

    '''
    def make_tf_dataset(self):
        print('making tf dataset')
        if self.parameters.get('model_auxiliary'):
            self.tf_train_dataset = tf.data.Dataset.from_tensor_slices(( \
                self.train_dataset_images_t, self.train_dataset_joints, self.train_dataset_cmd, \
                self.train_dataset_optical_flow, self.train_dataset_optical_flow, self.train_dataset_optical_flow, self.train_dataset_optical_flow \
                ))
            self.tf_test_dataset = tf.data.Dataset.from_tensor_slices(( \
                self.test_dataset_images_t, self.test_dataset_joints, self.test_dataset_cmd, \
                self.test_dataset_optical_flow, self.test_dataset_optical_flow, self.test_dataset_optical_flow, self.test_dataset_optical_flow \
                ))
        else:
            self.tf_train_dataset = tf.data.Dataset.from_tensor_slices(( \
                self.train_dataset_images_t, self.train_dataset_joints, self.train_dataset_cmd, \
                self.train_dataset_optical_flow \
                ))
            self.tf_test_dataset = tf.data.Dataset.from_tensor_slices(( \
                self.test_dataset_images_t, self.test_dataset_joints, self.test_dataset_cmd, \
                self.test_dataset_optical_flow \
                ))

        if self.parameters.get('dataset_shuffle'):
            self.tf_train_dataset = self.tf_train_dataset.shuffle(100000, reshuffle_each_iteration=True).batch(self.parameters.get('model_batch_size'), drop_remainder=True)
            self.tf_test_dataset = self.tf_test_dataset.batch(self.parameters.get('model_batch_size'))
        else:
            self.tf_train_dataset = self.tf_train_dataset.batch(self.parameters.get('model_batch_size'))
            self.tf_test_dataset = self.tf_test_dataset.batch(self.parameters.get('model_batch_size'))
    '''