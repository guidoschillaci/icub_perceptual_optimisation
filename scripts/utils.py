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

# the activation function of the output layer of the model
def activation_opt_flow(x):
    # apply activation
    x0 = K.relu(x[..., 0])  # magnitude is normalised between 0 and 1
    x1 = K.tanh(x[..., 1])  # cos(alpha) can have values between -1 and 1
    x2 = K.tanh(x[..., 2])  # sin(alpha) can have values between -1 and 1
    #out = [x0,x1,x2]
    return tf.stack((x0,x1,x2),axis=-1)

def sensory_attenuation(predicted_opt_flow, next_image, background_image):
    #amplified_pred_optflow = tf.math.sigmoid(predicted_opt_flow)
    #result = np.zeros((next_image.shape[0], next_image.shape[1], 3), np.uint8)
    unnorm_next = (next_image * 255.0).astype(np.uint8)
    result = np.multiply((1.0 - predicted_opt_flow/255), unnorm_next) + np.multiply(predicted_opt_flow/255, background_image)

    #print('max optflow', np.amax(predicted_opt_flow))
    #print('min optflow', np.amin(predicted_opt_flow))
    #print('max next_image', np.amax(next_image))
    #print('min next_image', np.amin(next_image))
    return result

class Split(tf.keras.layers.Layer):
    def __init__(self):
        super(Split, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, input, **kwargs):
        #return tf.split(input, 3, axis=1)
        return input[0],input[1],input[2]

class MyCallback(Callback):
    def __init__(self, param, datasets, model, model_pre_fusion, model_custom_fusion):
        self.parameters =param
        self.datasets = datasets
        #if self.parameters.get('model_custom_training_loop'):
        self.model = model
        self.model_pre_fusion = model_pre_fusion
        self.model_custom_fusion = model_custom_fusion
        #self.logs = {}

    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': []}
        # sub model with fusion weights output
        self.model_fusion_weights = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(name='fusion_weights').output)
        #if self.parameters.get('make_plots'):
        #    self.plot_predictions_test_dataset(-1, logs)
        #    # plot also sequences of predictions
        #    self.plot_train_sequences()

    def on_train_end(self, logs=None):
        if self.parameters.get('make_plots'):
            # plot also sequences of predictions
            self.plot_train_sequences(save_gif=True)
        #self.save_plots()

    def on_epoch_end(self, epoch, logs=None):
        #logs_keys = list(logs.keys())
        #print(logs_keys)
        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        #logs['loss'] =


    def plot_train_sequences(self, save_gif=False):
        start = [700, 1300, 3000, 3780, 4570, 5100, 7497, 11900]
        end = list(np.asarray(start) + self.parameters.get('plots_predict_size'))
        print('saving sequence plots...')
        for i in tqdm(range(len(start))):
            #print('plotting train '+str(i)+' of '+ str(len(start)) + ' ('  + str(start[i]) + ' to ' + str(end[i]) + ')')
            fusion_weights = self.plot_predictions('pred_sequence_train_' + str(start[i]) + '_' + str(end[i]), \
                                                   self.datasets.dataset_images_t[start[i]:end[i]], \
                                                   self.datasets.dataset_images_tp1[start[i]:end[i]], \
                                                   self.datasets.dataset_joints[start[i]:end[i]], \
                                                   self.datasets.dataset_cmd[start[i]:end[i]], \
                                                   self.datasets.dataset_optical_flow[start[i]:end[i]],\
                                                   save_gif=save_gif)

    #def get_fusion_weights(self):
    #    return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])

    def threshold_optical_flow(self, optflow):
        return np.where(optflow > self.parameters.get('opt_flow_binary_threshold'), 255, 0)

    def plot_predictions(self, filename, images_t, images_tp1, joints, commands, opt_flow, save_gif=False):
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
            plt.imshow(images_t[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')), cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_ylabel('img(t)', rotation=0)
            count_line = count_line +1

            ax2 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * self.parameters.get('plots_predict_size') + 1)
            plt.imshow(images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),cmap='gray')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.set_ylabel('img(t+1)', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_imgp1_'+str(i)+ '.png', bbox_inches=extent_2)
            count_line = count_line + 1

            ax3 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            opt_unnorm = deepcopy(opt_flow[i])
            if self.parameters.get('opt_flow_only_magnitude'):
                opt_unnorm = self.threshold_optical_flow(opt_unnorm)# * self.parameters.get('opt_flow_max_value'))
            else:
                opt_unnorm = self.threshold_optical_flow(opt_unnorm[...,0])# * self.parameters.get('opt_flow_max_value'))

            plt.imshow(opt_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
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
            pred_unnorm = deepcopy(predictions[i])
            #print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
            if self.parameters.get('opt_flow_only_magnitude'):
                pred_unnorm = self.threshold_optical_flow(pred_unnorm)# * self.parameters.get('opt_flow_max_value'))
            else:
                pred_unnorm = self.threshold_optical_flow(pred_unnorm[...,0] )# * self.parameters.get('opt_flow_max_value'))
            plt.imshow(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax4.get_xaxis().set_visible(False)
            ax4.get_yaxis().set_visible(False)
            ax4.set_ylabel('pred.OF', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_predOF_'+str(i)+ '.png', bbox_inches=extent_4)
            count_line = count_line + 1

            ax6 = plt.subplot(num_subplots, self.parameters.get('plots_predict_size'), i + count_line * (self.parameters.get('plots_predict_size')) + 1)
            attenuated_image_tp1=sensory_attenuation(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                                images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                                self.datasets.background_image)
            plt.imshow(attenuated_image_tp1, cmap='gray')
            ax6.get_xaxis().set_visible(False)
            ax6.get_yaxis().set_visible(False)
            ax6.set_ylabel('att.(t+1)', rotation=0)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_6 = ax6.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_attenuated_'+str(i)+ '.png', bbox_inches=extent_6)
            count_line = count_line + 1

            count_line = self.custom_weight_plots(0.6, 0.3, 0.1, images_t, deepcopy(images_tp1), joints, commands, num_subplots,
                                     i, count_line, bar_label, save_gif, fig, filename)
            count_line = self.custom_weight_plots(0.5, 0.3, 0.2, images_t, deepcopy(images_tp1), joints, commands,
                                                  num_subplots,
                                                  i, count_line, bar_label, save_gif, fig, filename)
            count_line = self.custom_weight_plots(0.3, 0.4, 0.3, images_t, deepcopy(images_tp1), joints, commands,
                                                  num_subplots,
                                                  i, count_line, bar_label, save_gif, fig, filename)
            count_line = self.custom_weight_plots(0.45, 0.1, 0.45, images_t, deepcopy(images_tp1), joints, commands,
                                                  num_subplots,
                                                  i, count_line, bar_label, save_gif, fig, filename)
            count_line = self.custom_weight_plots(0.2, 0.3, 0.5, images_t, deepcopy(images_tp1), joints, commands,
                                                  num_subplots,
                                                  i, count_line, bar_label, save_gif, fig, filename)
            count_line = self.custom_weight_plots(0.1, 0.3, 0.6, images_t, deepcopy(images_tp1), joints, commands, num_subplots,
                                     i, count_line, bar_label, save_gif, fig, filename)

            count_line = count_line + 1
        plt.savefig(self.parameters.get('directory_plots') + filename + '.png')

        return fusion_weights

    def custom_weight_plots(self, _wv,_wj,_wm, images_t, images_tp1, joints, commands, num_subplots, iter, count_line, bar_label, save_gif, fig, filename):
        w_v = np.ones(shape=[len(images_t),])*_wv
        w_j = np.ones(shape=[len(images_t),])*_wj
        w_m = np.ones(shape=[len(images_t),])*_wm
        pred_pre_fusion = self.model_pre_fusion.predict([images_t, joints, commands])
        #print(pred_pre_fusion[0].shape)
        pred_custom_fusion_allvision = self.model_custom_fusion.predict([pred_pre_fusion[0], w_v,
                                                                 pred_pre_fusion[1], w_j,
                                                                 pred_pre_fusion[2], w_m])

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
        predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[iter])
        # print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
        if self.parameters.get('opt_flow_only_magnitude'):
            predcustom_unnorm = self.threshold_optical_flow(predcustom_unnorm)
        else:
            predcustom_unnorm = self.threshold_optical_flow(predcustom_unnorm[..., 0])
        plt.imshow(predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                   cmap='gray')
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
        attenuated_custom = sensory_attenuation(
            predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
            images_tp1[iter].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
            self.datasets.background_image)
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
        self.plot_predictions('predictions_epoch_' + str(epoch),\
                                               self.datasets.test_dataset_images_t[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_images_tp1[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_joints[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_cmd[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_optical_flow[0:self.parameters.get('plots_predict_size')])

