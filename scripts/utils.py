from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import cv2
from tensorflow import keras as tfk
from utils_sim import arucoDetector

# the activation function of the output layer of the model
def activation_opt_flow(x):
    # apply activation
    x0 = K.relu(x[..., 0])  # magnitude is normalised between 0 and 1
    x1 = K.tanh(x[..., 1])  # cos(alpha) can have values between -1 and 1
    x2 = K.tanh(x[..., 2])  # sin(alpha) can have values between -1 and 1
    #out = [x0,x1,x2]
    return tf.stack((x0,x1,x2),axis=-1)

def sensory_attenuation(predicted_opt_flow, next_image, background_image, unnorm_tp1 = False):
    if unnorm_tp1:
        unnorm_next = (next_image * 255.0).astype(np.uint8)
        return np.multiply((1.0 - predicted_opt_flow/255), unnorm_next) + np.multiply(predicted_opt_flow/255, background_image)
    else:
        return np.multiply((1.0 - predicted_opt_flow/255), next_image) + np.multiply(predicted_opt_flow/255, background_image)

class Split(tf.keras.layers.Layer):
    def __init__(self):
        super(Split, self).__init__()

    def call(self, input, **kwargs):
        return tf.split(input, 3, axis=1)
        #return input[0],input[1],input[2]

class MyCallback(Callback):
    def __init__(self, param, datasets, model, model_pre_fusion, model_custom_fusion):
        self.parameters =param
        self.datasets = datasets
        self.model = model
        self.model_pre_fusion_features = model_pre_fusion
        self.model_custom_fusion = model_custom_fusion
        #self.logs = {}
        self.aruco_detector = arucoDetector()

        # list containing the custom fusion weights to be used in modulating predictions
        self.custom_weigths = []
        self.custom_weigths.append( (0.6, 0.3, 0.1) )
        self.custom_weigths.append( (0.5, 0.3, 0.2) )
        self.custom_weigths.append( (0.3, 0.4, 0.3) )
        self.custom_weigths.append( (0.45, 0.1, 0.45) )
        self.custom_weigths.append( (0.2, 0.3, 0.5) )
        self.custom_weigths.append( (0.1, 0.3, 0.6) )

    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': [], 'IoU': []}
        # sub model with fusion weights output
        self.model_fusion_weights = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(name='fusion_weights').output)
        #if self.parameters.get('make_plots'):
        #    # plot also sequences of predictions
        #    self.plot_train_sequences(save_gif=True)
        self.test_marker_detection()

    def on_train_end(self, logs=None):
        if self.parameters.get('make_plots'):
            # plot also sequences of predictions
            self.plot_train_sequences(save_gif=True)

    def on_epoch_end(self, epoch, logs=None):
        #logs_keys = list(logs.keys())
        #print(logs_keys)
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        self.history['IoU'].append(logs['val_IoU'])
        #logs['loss'] =

    # this is done on the original unshuffled dataset, because we want to show trajectories
    def plot_train_sequences(self, save_gif=False):
        start = [700, 1300, 3000, 3780, 4570, 5100]
        end = list(np.asarray(start) + self.parameters.get('plots_predict_size'))
        print('saving sequence plots...')
        for i in tqdm(range(len(start))):
            #print('plotting train '+str(i)+' of '+ str(len(start)) + ' ('  + str(start[i]) + ' to ' + str(end[i]) + ')')
            fusion_weights = self.plot_predictions('pred_sequence_train_' + str(start[i]) + '_' + str(end[i]), \
                                                   self.datasets.train_unshuffled.images_t[start[i]:end[i]], \
                                                   self.datasets.train_unshuffled.images_orig_size_t[start[i]:end[i]], \
                                                   self.datasets.train_unshuffled.images_orig_size_tp1[start[i]:end[i]], \
                                                   self.datasets.train_unshuffled.joints[start[i]:end[i]], \
                                                   self.datasets.train_unshuffled.cmd[start[i]:end[i]], \
                                                   self.datasets.train_unshuffled.optical_flow[start[i]:end[i]],\
                                                   save_gif=save_gif)

    #def get_fusion_weights(self):
    #    return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])

    # output image has values: 0 or positive_value
    def binarize_optical_flow(self, optflow, positive_value = 255):
        if self.parameters.get('opt_flow_binarize'):
            return np.array(np.where(optflow > self.parameters.get('opt_flow_binary_threshold'), positive_value, 0), dtype='uint8')
        return np.array(optflow, dtype='uint8')

    def plot_predictions(self, filename, images_t, images_t_orig_size, images_tp1_orig_size, joints, commands, opt_flow, save_gif=False):
        predictions_all_outputs = self.model.predict([images_t, joints, commands])
        if self.parameters.get('model_auxiliary'):
            predictions = predictions_all_outputs[0]
        else:
            predictions = predictions_all_outputs

        fusion_weights = self.model_fusion_weights.predict([images_t, \
                                                            joints, \
                                                            commands])
        bar_label = ['v', 'p', 'm']
        num_subplots = 25

        fig = plt.figure(figsize=(6, 20))
        for i in range(self.parameters.get('plots_predict_size')):
            count_line = 0
            # display original
            ax1 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * self.parameters.get('plots_predict_size') +1)
            #plt.imshow(images_t[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')), cmap='gray')
            plt.imshow(cv2.resize(images_t_orig_size[i], self.parameters.get('image_original_shape')), cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_ylabel('img(t)', rotation=0)
            count_line = count_line +1

            ax2 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * self.parameters.get('plots_predict_size') + 1)
            #plt.imshow(images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),cmap='gray')
            plt.imshow(cv2.resize(images_tp1_orig_size[i], self.parameters.get('image_original_shape')), cmap='gray')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.set_ylabel('img(t+1)', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_imgp1_'+str(i)+ '.png', bbox_inches=extent_2)
            count_line = count_line + 1

            ax3 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            opt_unnorm = deepcopy(opt_flow[i].squeeze())
            #if self.parameters.get('opt_flow_only_magnitude'):
            opt_unnorm = self.binarize_optical_flow(opt_unnorm)# * self.parameters.get('opt_flow_max_value'))
            #else:
            #    opt_unnorm = self.binarize_optical_flow(opt_unnorm[..., 0])# * self.parameters.get('opt_flow_max_value'))

            #plt.imshow(opt_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
            #           cmap='gray')
            cv2_opt_unnorm = cv2.resize(opt_unnorm, self.parameters.get('image_original_shape'))
            plt.imshow(cv2_opt_unnorm, cmap='gray')
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            ax3.set_ylabel('true OF', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_trueOF_'+str(i)+ '.png', bbox_inches=extent_3)
            count_line = count_line + 1

            ax5 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), \
                              i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            ax5.set_ylim(0, 1)
            plt.bar(bar_label, fusion_weights[i], width=0.3)
            ax5.set_ylabel('fus. w.', rotation=0)
            if i != 0:
                ax5.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_5 = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_fw_'+str(i)+ '.png', bbox_inches=extent_5)
            count_line = count_line + 1

            ax4 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            pred_unnorm = deepcopy(predictions[i].squeeze())
            #pred_unnorm = pred_unnorm.reshape(self.parameters.get('image_original_shape'))
            #print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
            #if self.parameters.get('opt_flow_only_magnitude'):
            pred_unnorm = self.binarize_optical_flow(pred_unnorm)# * self.parameters.get('opt_flow_max_value'))
            #else:
            #    pred_unnorm = self.binarize_optical_flow(pred_unnorm[..., 0])# * self.parameters.get('opt_flow_max_value'))
            #plt.imshow(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
            #           cmap='gray')
            cv2_pred_unnorm = cv2.resize(pred_unnorm, self.parameters.get('image_original_shape'))
            plt.imshow(cv2_pred_unnorm, cmap='gray')
            ax4.get_xaxis().set_visible(False)
            ax4.get_yaxis().set_visible(False)
            ax4.set_ylabel('pred.OF', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_predOF_'+str(i)+ '.png', bbox_inches=extent_4)
            count_line = count_line + 1

            ax6 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            #attenuated_image_tp1=sensory_attenuation(pred_unnorm.reshape(self.parameters.get('image_original_shape')),
            #                    images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
            #                    self.datasets.background_image)
            #print('cv2_pred_unnorm shape ', cv2_pred_unnorm.shape)
            #print('images_tp1_orig_size[i] shape ', images_tp1_orig_size[i].shape)
            #print('self.datasets.train.background_image shape ', self.datasets.train.background_image.shape)
            _images_tp1_orig_size = cv2.resize(images_tp1_orig_size[i], self.parameters.get('image_original_shape'))
            attenuated_image_tp1 = sensory_attenuation(cv2_pred_unnorm, _images_tp1_orig_size, self.datasets.train.background_image)
            plt.imshow(attenuated_image_tp1, cmap='gray')
            ax6.get_xaxis().set_visible(False)
            ax6.get_yaxis().set_visible(False)
            ax6.set_ylabel('att.(t+1)', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_6 = ax6.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_attenuated_'+str(i)+ '.png', bbox_inches=extent_6)
            count_line = count_line + 1

            for cw in range(len(self.custom_weigths)):
                count_line = self.custom_weight_plots(self.custom_weigths[cw], images_t, deepcopy(images_t_orig_size), deepcopy(images_tp1_orig_size), \
                                                  joints, commands, \
                                                  num_subplots, i, count_line, bar_label, save_gif, fig, filename)
            count_line = count_line + 1
        plt.savefig(self.parameters.get('directory_plots') + filename + '.png')

        return fusion_weights

    def custom_weight_plots(self, custom_weights, images_t, images_t_orig_size, images_tp1_orig_size, joints, commands, num_subplots, iter, count_line, bar_label, save_gif, fig, filename):
        w_v = np.ones(shape=[len(images_t),])*custom_weights[0]
        w_j = np.ones(shape=[len(images_t),])*custom_weights[1]
        w_m = np.ones(shape=[len(images_t),])*custom_weights[2]
        pred_pre_fusion_features = self.model_pre_fusion_features([images_t, joints, commands], training=False)
        #print(pred_pre_fusion[0].shape)
        pred_custom_fusion_allvision = self.model_custom_fusion([pred_pre_fusion_features[0], w_v, w_v,
                                                                 pred_pre_fusion_features[1], w_j, w_j,
                                                                 pred_pre_fusion_features[2], w_m, w_m], training=False)

        ax7 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), \
                          iter + count_line * (self.parameters.get('plots_predict_size')) + 1)
        ax7.set_ylim(0, 1)
        plt.bar(bar_label, [w_v[0], w_j[0], w_m[0]], width=0.3)
        ax7.set_ylabel('fus. w', rotation=0)
        if iter != 0:
            ax7.get_yaxis().set_visible(False)
        if save_gif:
            # Save just the portion _inside_ the second axis's boundaries
            extent_7 = ax7.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_custom_fw_' + str(iter) + '.png',
                        bbox_inches=extent_7)
        count_line = count_line + 1

        ax8 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'),
                          iter + count_line * (self.parameters.get('plots_predict_size')) + 1)
        predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[iter].numpy())
        #predcustom_unnorm = predcustom_unnorm.reshape(self.parameters.get('image_original_shape'))
        # print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
        #if self.parameters.get('opt_flow_only_magnitude'):
        predcustom_unnorm = self.binarize_optical_flow(predcustom_unnorm)
        #else:
        #    predcustom_unnorm = self.binarize_optical_flow(predcustom_unnorm[..., 0])
        #plt.imshow(predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
        #           cmap='gray')
        cv2_predcustom_unnorm = cv2.resize(predcustom_unnorm, self.parameters.get('image_original_shape'))

        plt.imshow(cv2_predcustom_unnorm, cmap='gray')
        ax8.get_xaxis().set_visible(False)
        ax8.get_yaxis().set_visible(False)
        ax8.set_ylabel('custom pred.', rotation=0)
        if save_gif:
            # Save just the portion _inside_ the second axis's boundaries
            extent_8 = ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_custom_predOF_' + str(iter) + '.png',
                        bbox_inches=extent_8)
        count_line = count_line + 1

        ax9 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'),
                          iter + count_line * (self.parameters.get('plots_predict_size')) + 1)
        #attenuated_custom = sensory_attenuation(
        #    predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
        #    images_tp1[iter].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
        #    self.datasets.background_image)
        _images_tp1_orig_size = cv2.resize(images_tp1_orig_size[iter], self.parameters.get('image_original_shape'))
        attenuated_custom = sensory_attenuation(cv2_predcustom_unnorm, _images_tp1_orig_size, self.datasets.train.background_image)
        plt.imshow(attenuated_custom, cmap='gray')
        ax9.get_xaxis().set_visible(False)
        ax9.get_yaxis().set_visible(False)
        if save_gif:
            # Save just the portion _inside_ the second axis's boundaries
            extent_9 = ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_attenuated_custom_' + str(iter) + '.png',
                        bbox_inches=extent_9)
        count_line = count_line + 1
        return count_line

    def plot_predictions_test_dataset(self, epoch, logs):
        print('Callback: saving predicted images')
        self.plot_predictions('predictions_epoch_' + str(epoch), \
                              self.datasets.test.images_t[0:self.parameters.get('plots_predict_size')], \
                              self.datasets.test.images_orig_size_t[0:self.parameters.get('plots_predict_size')], \
                              self.datasets.test.images_orig_size_tp1[0:self.parameters.get('plots_predict_size')], \
                              self.datasets.test.joints[0:self.parameters.get('plots_predict_size')], \
                              self.datasets.test.cmd[0:self.parameters.get('plots_predict_size')], \
                              self.datasets.test.optical_flow[0:self.parameters.get('plots_predict_size')])

    # attenuate on test dataset using learned weights
    def attenuate_test_ds(self):
        predictions_all_outputs = self.model.predict([self.datasets.test.images_t, \
                                                      self.datasets.test.joints, \
                                                      self.datasets.test.cmd])
        if self.parameters.get('model_auxiliary'):
            predictions = predictions_all_outputs[0]
        else:
            predictions = predictions_all_outputs

        attenuated_imgs = []
        for i in range(len(predictions)):
            pred_unnorm = predictions[i].squeeze()
            cv2_pred_unnorm = cv2.resize(pred_unnorm, self.parameters.get('image_original_shape'))
            # binarise pred_unnorm (has values 0 or 255)
            #if self.parameters.get('opt_flow_only_magnitude'):
            cv2_pred_unnorm = self.binarize_optical_flow(
                cv2_pred_unnorm)  # * self.parameters.get('opt_flow_max_value'))
            #else:
            #    cv2_pred_unnorm = self.binarize_optical_flow(cv2_pred_unnorm[..., 0])
            _images_orig_size_tp1 = cv2.resize(self.datasets.test.images_orig_size_tp1[i], self.parameters.get('image_original_shape'))
            attenuated_image_tp1 = sensory_attenuation(cv2_pred_unnorm,
                                                       _images_orig_size_tp1,
                                                       self.datasets.train.background_image)
            attenuated_imgs.append(attenuated_image_tp1)
        return attenuated_imgs

    # attenuate on test dataset using custom weights
    def attenuate_test_ds_using_custom_weights(self, custom_weights):
        len_img = len(self.datasets.test.images_orig_size_t)
        w_v = np.ones(shape=[len_img,])*custom_weights[0]
        w_j = np.ones(shape=[len_img,])*custom_weights[1]
        w_m = np.ones(shape=[len_img,])*custom_weights[2]
        # get the features extracted from each modality before the fusion
        pred_pre_fusion_features = \
            self.model_pre_fusion_features([self.datasets.test.images_t, \
                                            self.datasets.test.joints, \
                                            self.datasets.test.cmd], training=False)
        # predict doing the fusion with the custom fusion weights
        pred_custom_fusion_allvision = \
            self.model_custom_fusion([pred_pre_fusion_features[0], w_v, w_v, \
                                      pred_pre_fusion_features[1], w_j, w_j, \
                                      pred_pre_fusion_features[2], w_m, w_m], training=False)

        attenuated_imgs = []
        for i in range(len(pred_custom_fusion_allvision)):
            predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[i].numpy())
            #if self.parameters.get('opt_flow_only_magnitude'):
            predcustom_unnorm = self.binarize_optical_flow(predcustom_unnorm)
            #else:
            #    predcustom_unnorm = self.binarize_optical_flow(predcustom_unnorm[..., 0])
            cv2_predcustom_unnorm = cv2.resize(predcustom_unnorm, self.parameters.get('image_original_shape'))

            _images_orig_size_tp1 = cv2.resize(self.datasets.test.images_orig_size_tp1[i], self.parameters.get('image_original_shape'))
            attenuated_img = sensory_attenuation(cv2_predcustom_unnorm, \
                                                    _images_orig_size_tp1, \
                                                    self.datasets.train.background_image)
            attenuated_imgs.append(attenuated_img)
        return attenuated_imgs

    def test_marker_detection(self):
        print('testing marker detection...')
        # counting markers in original images
        self.results_markers_in_orig_img = \
            self.aruco_detector.avg_mrk_in_list_of_img(self.datasets.test.images_orig_size_tp1)
        print('average markers in original images: ' + str(self.results_markers_in_orig_img))
        # count markers in the imgs where sensory attenuation is perfomed

        ## first, predicting optflows using the learned fusion weights
        attenuated_imgs_using_learned_weights = self.attenuate_test_ds()
        self.results_markers_in_attenuated_img = \
            self.aruco_detector.avg_mrk_in_list_of_img(attenuated_imgs_using_learned_weights)
        print('average markers in attenuated images: ' + str(self.results_markers_in_attenuated_img))

        ## then, using custom weights
        self.results_markers_in_attenuated_img_with_custom_weights = []
        # for each set of custom weights
        for i in range(len(self.custom_weigths)):
            attenuated_imgs_with_custom_weights = \
                self.attenuate_test_ds_using_custom_weights(self.custom_weigths[i])
            res = self.aruco_detector.avg_mrk_in_list_of_img(attenuated_imgs_with_custom_weights)
            self.results_markers_in_attenuated_img_with_custom_weights.append(res)
            print('average markers in attenuated images with custom weights (set '+str(i)+'): ' \
                  + str(res))
