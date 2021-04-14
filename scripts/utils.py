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

# output image has values: 0 or positive_value
def binarize_optical_flow(param, optflow, positive_value = 255):
    if param.get('opt_flow_binarize'):
        return np.array(np.where(optflow > param.get('opt_flow_binary_threshold'), positive_value, 0), dtype='uint8')
    return np.array(optflow, dtype='uint8')

def intersection_over_union(param, y_true, y_pred, is_true_binarised, is_pred_binarised):
    if not is_true_binarised:
        y_true_binarised = binarize_optical_flow(param, y_true, positive_value=1)
    else:
        y_true_binarised = y_true / np.max(y_true) # positive value was 255
    if not is_pred_binarised:
        y_pred_binarised = binarize_optical_flow(param, y_pred, positive_value=1)
    else:
        y_pred_binarised = y_pred / np.max(y_pred)

    print('y_true shape ', np.asarray(y_true_binarised).shape, ' max ', np.max(y_true_binarised), ' min ',
          np.min(y_true_binarised))
    print('y_pred shape ', np.asarray(y_pred_binarised).shape, ' max ', np.max(y_pred_binarised), ' min ',
          np.min(y_pred_binarised))
    intersection = np.multiply(y_true_binarised, y_pred_binarised)
    union = y_true_binarised + y_pred_binarised - intersection
    count_intersection = np.count_nonzero(intersection)
    count_union = np.count_nonzero(union)
    print('int shape ', np.asarray(intersection).shape)
    print('union shape ', np.asarray(union).shape)
    print('count int ', count_intersection)

    print('count uni ', count_union)
    print('iou ', count_intersection / count_union)
    cv2.imwrite(param.get('directory_plots')+'y_pred_bin.png', y_pred_binarised*255)
    cv2.imwrite(param.get('directory_plots')+'y_true_bin.png', y_true_binarised*255)
    return count_intersection / count_union

# output image has values: 0 or positive_value
def tf_binarize_optical_flow(param, optflow, positive_value = 255):
    if param.get('opt_flow_binarize'):
        return tf.cast(tf.where(tf.greater(optflow, param.get('opt_flow_binary_threshold')), positive_value, 0), tf.uint8)
    return tf.cast(optflow, tf.uint8)

def tf_intersection_over_union(param, y_true, y_pred):
    y_true_binarised = tf_binarize_optical_flow(param, y_true, positive_value=1)
    y_pred_binarised = tf_binarize_optical_flow(param, y_pred, positive_value=1)
    intersection = tf.math.multiply(y_true_binarised, y_pred_binarised)
    union = y_true_binarised + y_pred_binarised - intersection
    count_intersection = tf.math.count_nonzero(intersection)
    count_union = tf.math.count_nonzero(union)
    return count_intersection / count_union

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

        # these will contain marker detection results
        self.markers_in_orig_img = []
        self.markers_in_attenuated_img = []  # predictions executed with main model
        self.markers_in_attenuated_img_with_custom_weights = []
        for i in range(len(self.custom_weigths)):
            self.markers_in_attenuated_img_with_custom_weights.append([])

        # these will contain intersection over units results
        self.iou_main_model = []  # predictions executed with main model
        self.iou_model_with_custom_weights = []
        for i in range(len(self.custom_weigths)):
            self.iou_model_with_custom_weights.append([])


    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': []}#, 'IoU': []}
        # sub model with fusion weights output
        self.model_fusion_weights = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(name='fusion_weights').output)
        #if self.parameters.get('make_plots'):
        #    # plot also sequences of predictions
        #    self.plot_train_sequences(save_gif=True)
        self.test_marker_detection()

    def on_train_end(self, logs=None):
        self.save_marker_data()
        self.save_iou_data()
        if self.parameters.get('make_plots'):
            # plot also sequences of predictions
            self.plot_train_sequences(save_gif=self.parameters.get('save_sequence_plots_gif'))

    def on_epoch_end(self, epoch, logs=None):
        #logs_keys = list(logs.keys())
        #print(logs_keys)
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        #self.history['IoU'].append(logs['val_IoU'])
        self.test_marker_detection()
        #logs['loss'] =

    # this is done on the original unshuffled dataset, because we want to show trajectories
    def plot_train_sequences(self, save_gif=False):
        #start = [int(len(self.datasets.test.images_t) * 0.1), \
        #         int(len(self.datasets.test.images_t) * 0.2), \
        #         int(len(self.datasets.test.images_t) * 0.3), \
        #         int(len(self.datasets.test.images_t) * 0.4), \
        #         int(len(self.datasets.test.images_t) * 0.5), \
        #         int(len(self.datasets.test.images_t) * 0.6), \
        #         int(len(self.datasets.test.images_t) * 0.7), \
        #         int(len(self.datasets.test.images_t) * 0.8), \
        #         ]
        start = [int(len(self.datasets.test.images_t) * 0.1), \
                 int(len(self.datasets.test.images_t) * 0.3), \
                 int(len(self.datasets.test.images_t) * 0.5), \
                 int(len(self.datasets.test.images_t) * 0.7), \
                 ]
        end = list(np.asarray(start) + self.parameters.get('plots_predict_size'))
        if self.parameters.get('save_sequence_plots'):
            print('saving sequence plots...')
            for i in tqdm(range(len(start))):
                #print('plotting train '+str(i)+' of '+ str(len(start)) + ' ('  + str(start[i]) + ' to ' + str(end[i]) + ')')
                fusion_weights = self.plot_predictions('pred_sequence_train_' + str(start[i]) + '_' + str(end[i]), \
                                                       self.datasets.test.images_t[start[i]:end[i]], \
                                                       self.datasets.test.images_orig_size_t[start[i]:end[i]], \
                                                       self.datasets.test.images_orig_size_tp1[start[i]:end[i]], \
                                                       self.datasets.test.joints[start[i]:end[i]], \
                                                       self.datasets.test.cmd[start[i]:end[i]], \
                                                       self.datasets.test.optical_flow[start[i]:end[i]],\
                                                       save_gif=save_gif)

    #def get_fusion_weights(self):
    #    return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])


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
        horiz_fig_size = 6 * int(self.parameters.get('plots_predict_size') / 5)
        fig = plt.figure(figsize=(horiz_fig_size, 20))
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
            opt_unnorm = binarize_optical_flow(self.parameters, opt_unnorm)# * self.parameters.get('opt_flow_max_value'))
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
            pred_unnorm = binarize_optical_flow(self.parameters, pred_unnorm)# * self.parameters.get('opt_flow_max_value'))
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
                count_line = self.custom_weight_plots(cw, self.custom_weigths[cw], images_t, deepcopy(images_t_orig_size), deepcopy(images_tp1_orig_size), \
                                                  joints, commands, \
                                                  num_subplots, i, count_line, bar_label, save_gif, fig, filename)
            count_line = count_line + 1
        plt.savefig(self.parameters.get('directory_plots') + filename + '.png')

        return fusion_weights

    def custom_weight_plots(self, custom_weights_id, custom_weights, images_t, images_t_orig_size, images_tp1_orig_size, joints, commands, num_subplots, iter, count_line, bar_label, save_gif, fig, filename):
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
        plt.bar(bar_label, [w_v[iter], w_j[iter], w_m[iter]], width=0.3)
        ax7.set_ylabel('fus. w', rotation=0)

        ax7.set_xticks(['v', 'p', 'm'])
        #ax7.get_xticklabels().set_visible(True)
        if iter != 0:
            ax7.get_yaxis().set_visible(False)
        if save_gif:
            # Save just the portion _inside_ the second axis's boundaries
            extent_7 = ax7.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_custom_fw_' + str(custom_weights_id) \
                        + '_' + str(iter) + '.png',
                        bbox_inches=extent_7)
        count_line = count_line + 1

        ax8 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'),
                          iter + count_line * (self.parameters.get('plots_predict_size')) + 1)
        predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[iter].numpy())
        #predcustom_unnorm = predcustom_unnorm.reshape(self.parameters.get('image_original_shape'))
        # print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
        #if self.parameters.get('opt_flow_only_magnitude'):
        predcustom_unnorm = binarize_optical_flow(self.parameters, predcustom_unnorm)
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
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_custom_predOF_' +str(custom_weights_id) \
                        + '_' + str(iter) + '.png', bbox_inches=extent_8)
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
            fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_attenuated_custom_' + str(custom_weights_id)  \
                + '_' + str(iter) + '.png', bbox_inches=extent_9)
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
        iou = []
        for i in range(len(predictions)):
            pred_unnorm = predictions[i].squeeze()

            iou.append(intersection_over_union(self.parameters, \
                                               self.datasets.test.optical_flow[i].squeeze(), \
                                               pred_unnorm, False, False))

            cv2_pred_unnorm = cv2.resize(pred_unnorm, self.parameters.get('image_original_shape'))
            # binarise pred_unnorm (has values 0 or 255)
            #if self.parameters.get('opt_flow_only_magnitude'):
            cv2_pred_unnorm = binarize_optical_flow(self.parameters,cv2_pred_unnorm)  # * self.parameters.get('opt_flow_max_value'))

            #else:
            #    cv2_pred_unnorm = self.binarize_optical_flow(cv2_pred_unnorm[..., 0])
            _images_orig_size_tp1 = cv2.resize(self.datasets.test.images_orig_size_tp1[i], self.parameters.get('image_original_shape'))

            attenuated_image_tp1 = sensory_attenuation(cv2_pred_unnorm,
                                                       _images_orig_size_tp1,
                                                       self.datasets.train.background_image)
            attenuated_imgs.append(attenuated_image_tp1)
        return attenuated_imgs, np.mean(np.asarray(iou))

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
        iou = []
        for i in range(len(pred_custom_fusion_allvision)):
            predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[i].numpy())

            iou.append(intersection_over_union(self.parameters, \
                                               self.datasets.test.optical_flow[i].squeeze(), \
                                               predcustom_unnorm, False, False))

            #if self.parameters.get('opt_flow_only_magnitude'):
            #else:
            #    predcustom_unnorm = self.binarize_optical_flow(predcustom_unnorm[..., 0])
            cv2_predcustom_unnorm = cv2.resize(predcustom_unnorm, self.parameters.get('image_original_shape'))
            cv2_predcustom_unnorm = binarize_optical_flow(self.parameters, cv2_predcustom_unnorm)

            _images_orig_size_tp1 = cv2.resize(self.datasets.test.images_orig_size_tp1[i], self.parameters.get('image_original_shape'))

            attenuated_img = sensory_attenuation(cv2_predcustom_unnorm, \
                                                    _images_orig_size_tp1, \
                                                    self.datasets.train.background_image)
            attenuated_imgs.append(attenuated_img)
        return attenuated_imgs, np.mean(np.asarray(iou))


    def test_marker_detection(self):
        print('testing marker detection...')
        #print('self.datasets.test.images_orig_size_tp1 shape', np.asarray(self.datasets.test.images_orig_size_tp1).shape)
        # counting markers in original images
        self.markers_in_orig_img.append( \
            self.aruco_detector.avg_mrk_in_list_of_img(self.datasets.test.images_orig_size_tp1) )
        print('average markers in original images: ' + str(self.markers_in_orig_img[-1]))
        # count markers in the imgs where sensory attenuation is perfomed

        ## first, predicting optflows using the learned fusion weights
        attenuated_imgs_using_learned_weights, iou_main = self.attenuate_test_ds()
        self.iou_main_model.append(iou_main)
        self.markers_in_attenuated_img.append( \
            self.aruco_detector.avg_mrk_in_list_of_img(attenuated_imgs_using_learned_weights))
        print('average markers in attenuated images: ' + str(self.markers_in_attenuated_img[-1]) + ' iou ' + str(iou_main))

        ## then, using custom weights
        #self.results_markers_in_attenuated_img_with_custom_weights = []
        # for each set of custom weights
        for i in range(len(self.custom_weigths)):
            attenuated_imgs_with_custom_weights, iou_custom_w = \
                self.attenuate_test_ds_using_custom_weights(self.custom_weigths[i])
            res = self.aruco_detector.avg_mrk_in_list_of_img(attenuated_imgs_with_custom_weights)
            self.markers_in_attenuated_img_with_custom_weights[i].append(res)
            self.iou_model_with_custom_weights[i].append(iou_custom_w)
            print('average markers in attenuated images with custom weights (set '+str(i)+'): ' \
                  + str(res) + ' iou ' + str(iou_custom_w) )

    def save_marker_data(self):
        print('saving marker detection results...')
        #res = []
        #res.append(self.results_markers_in_orig_img)
        #res.append(self.results_markers_in_attenuated_img)
        #for i in range(len(self.custom_weigths)):
        #    res.append(self.results_markers_in_attenuated_img_with_custom_weights[i])
        np.savetxt(self.parameters.get('directory_plots') + 'markers_in_original_img.txt', \
                   np.asarray(self.markers_in_orig_img), fmt="%s")
        np.savetxt(self.parameters.get('directory_plots') + 'markers_in_attenuated_img.txt', \
                   np.asarray(self.markers_in_attenuated_img), fmt="%s")
        for i in range(len(self.custom_weigths)):
            np.savetxt(self.parameters.get('directory_plots') + 'markers_in_att_custom_weig_'+str(i) + '.txt',
                   np.asarray(self.markers_in_attenuated_img_with_custom_weights[i]), fmt="%s")
        print('...saved')


    def save_iou_data(self):
        print('saving intersection_over_unit results...')
        #res = []
        #res.append(self.results_markers_in_orig_img)
        #res.append(self.results_markers_in_attenuated_img)
        #for i in range(len(self.custom_weigths)):
        #    res.append(self.results_markers_in_attenuated_img_with_custom_weights[i])
        np.savetxt(self.parameters.get('directory_plots') + 'iou_main_model.txt', \
                   np.asarray(self.iou_main_model), fmt="%s")
        for i in range(len(self.custom_weigths)):
            np.savetxt(self.parameters.get('directory_plots') + 'iou_model_with_custom_weights_'+str(i) + '.txt',
                   np.asarray(self.iou_model_with_custom_weights[i]), fmt="%s")
        print('...saved')