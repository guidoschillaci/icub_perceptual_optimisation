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

# the activation function of the output layer of the model
def activation_opt_flow(x):
    # apply activation
    x0 = K.relu(x[..., 0])  # magnitude is normalised between 0 and 1
    x1 = K.tanh(x[..., 1])  # cos(alpha) can have values between -1 and 1
    x2 = K.tanh(x[..., 2])  # sin(alpha) can have values between -1 and 1
    #out = [x0,x1,x2]
    return tf.stack((x0,x1,x2),axis=-1)

def sensory_attenuation(predicted_opt_flow, next_image, background_image):
    amplified_pred_optflow = tf.math.sigmoid(predicted_opt_flow)
    #result = np.zeros((next_image.shape[0], next_image.shape[1], 3), np.uint8)
    unnorm_next = (next_image * 255.0).astype(np.uint8)
    result = np.multiply((1.0 - amplified_pred_optflow), unnorm_next) + np.multiply(amplified_pred_optflow, background_image)

    return result
    #print('max optflow', np.amax(predicted_opt_flow))
    #print('min optflow', np.amin(predicted_opt_flow))
    #print('max next_image', np.amax(next_image))
    #print('min next_image', np.amin(next_image))

    #print('max background_image', np.amax(background_image))
    #print('min background_image', np.amin(background_image))


    #result = np.multiply(predicted_opt_flow, background_image).astype(np.uint8)
    #result[:, :, 1] = np.multiply((1. - predicted_opt_flow), next_image[:, :, 1]) + np.multiply(predicted_opt_flow, background_image[:, :, 1])
    #result[:, :, 2] = np.multiply((1. - predicted_opt_flow), next_image[:, :, 2]) + np.multiply(predicted_opt_flow, background_image[:, :, 2])
    #return result

    #unnorm_pred = (np.zeros(next_image.shape, dtype=np.uint8) + (1.0 - predicted_opt_flow)*255).astype(np.uint8)
    #unnorm_next = (next_image * 255).astype(np.uint8)
    # convert grayscale img to 3-channles + alpha
    #attenuated_image = cv2.merge((unnorm_next,unnorm_next,unnorm_next,unnorm_pred))
    #return np.uint8(cv2.addWeighted(next_image.astype(np.uint8), 0.5, result, 0.5, 0.0), dtype=np.uint8)

class Split(tf.keras.layers.Layer):
    def __init__(self):
        super(Split, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, input, **kwargs):
        return tf.split(input, 3, axis=1)

class MyCallback(Callback):
    def __init__(self, param, datasets, model, model_pre_fusion, model_custom_fusion):
        self.parameters =param
        self.datasets = datasets
        #if self.parameters.get('model_custom_training_loop'):
        self.model = model
        self.model_pre_fusion = model_pre_fusion
        self.model_custom_fusion = model_custom_fusion
        self.logs = {}

    def on_train_begin(self, logs={}):
        #print('log key', str(logs.keys()))

        #if not self.parameters.get('model_auxiliary'):
        self.history = {'loss': [], 'val_loss': []}
        #else:
        #    self.history = {'loss': [], \
        #                    'main_output_loss': [], \
        #                    'aux_visual_output_loss': [], \
        #                    'aux_proprio_output_loss': [], \
        #                    'aux_motor_output_loss': [], \
        #                    'val_loss': [], \
        #                    'val_main_output_loss': [], \
        #                    'val_aux_visual_output_loss': [], \
        #                    'val_aux_proprio_output_loss': [], \
        #                    'val_aux_motor_output_loss': []}

        #print('callback train begin')
        #print('train size ', str(len(self.datasets.dataset_images_t[[self.datasets.train_indexes]])))
        #print('teest size ', str(len(self.datasets.dataset_images_t[[self.datasets.test_indexes]])))
        # sub model with fusion weights output
        self.model_fusion_weights = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(name='fusion_weights').output)

        if self.parameters.get('make_plots'):
            self.plot_predictions_test_dataset(-1, logs)
            # plot also sequences of predictions
            self.plot_train_sequences()

    def on_batch_end (self, batch, logs={}):
        pass

    def on_train_end(self, logs=None):
        #print('callback train end')
        if self.parameters.get('make_plots'):
            # plot also sequences of predictions
            self.plot_train_sequences(save_gif=True)

        self.save_plots()

    def on_epoch_end(self, epoch, logs=None):
        #print('callback epoch end')
        #print('log key', str(logs.keys()))
        #print('hostiry key', str(self.history.keys()))
        #if not self.parameters.get('model_auxiliary'):
        self.history['loss'].append(self.logs.get('loss'))
        self.history['val_loss'].append(self.logs.get('val_loss'))
        #else:
        #    self.history['loss'].append(self.logs.get('loss'))
        #    self.history['main_output_loss'].append(self.logs.get('main_output_loss'))
        #    self.history['aux_visual_output_loss'].append(self.logs.get('aux_visual_output_loss'))
        #    self.history['aux_proprio_output_loss'].append(self.logs.get('aux_proprio_output_loss'))
        #    self.history['aux_motor_output_loss'].append(self.logs.get('aux_motor_output_loss'))

        #    self.history['val_loss'].append(self.logs.get('val_loss'))
        #    self.history['val_main_output_loss'].append(self.logs.get('val_main_output_loss'))
        #    self.history['val_aux_visual_output_loss'].append(self.logs.get('val_aux_visual_output_loss'))
        #    self.history['val_aux_proprio_output_loss'].append(self.logs.get('val_aux_proprio_output_loss'))
        #    self.history['val_aux_motor_output_loss'].append(self.logs.get('val_aux_motor_output_loss'))

        #if self.parameters.get('make_plots'):
        #    # plot also sequences of predictions
        #    self.plot_train_sequences(save_gif=True)
        #    self.plot_predictions_test_dataset(epoch, logs, predict_size=self.parameters.get('plots_predict_size'))

    def save_plots(self):
        pd.DataFrame.from_dict(self.history).to_csv(self.parameters.get('directory_results') +'history.csv', index=False)
        # history dictioary
        history_keys = list(self.history.keys())
        #print('keras history keys ', history_keys)

        # summarize history for loss
        fig = plt.figure(figsize=(10, 12))
        plt.title('model history')
        plt.ylabel('value')
        plt.xlabel('epoch')
        for i in range(len(history_keys)):
            #if (history_keys[i] == 'loss') or (history_keys[i]=='val_loss'):
            plt.plot(self.history[history_keys[i]], label=history_keys[i])
            np.savetxt(self.parameters.get('directory_plots') + history_keys[i]+ '.txt', self.history[history_keys[i]],fmt="%s")
        plt.legend(history_keys, loc='upper left')
        plt.savefig(self.parameters.get('directory_plots') + 'history.png')

        if self.parameters.get('model_auxiliary'):
            fig2 = plt.figure(figsize=(10, 10))
            plt.title('model history')
            plt.ylabel('value')
            plt.xlabel('epoch')
            for i in range(len(history_keys)):
                if (history_keys[i] == 'loss') or (history_keys[i] == 'val_loss'):
                    pass
                else:
                    plt.plot(self.history[history_keys[i]], label=history_keys[i])
                    np.savetxt(self.parameters.get('directory_plots') + history_keys[i] + '.txt',
                               self.history[history_keys[i]], fmt="%s")
            plt.legend(history_keys, loc='upper left')
            plt.savefig(self.parameters.get('directory_plots') + 'history_sub_losses.png')
        #plt.show()

    def plot_train_sequences(self, save_gif=False):
        start = [700, 1300, 3600, 3780, 4570, 5100, 7500, 13900]
        end = list(np.asarray(start) + self.parameters.get('plots_predict_size'))
        print('saving sequence plots...')
        for i in tqdm(range(len(start))):
            #print('plotting train '+str(i)+' of '+ str(len(start)) + ' ('  + str(start[i]) + ' to ' + str(end[i]) + ')')
            fusion_weights = self.plot_predictions('pred_sequence_train_' + str(start[i]) + '_' + str(end[i]), \
                                                   self.datasets.train_dataset_images_t[start[i]:end[i]], \
                                                   self.datasets.train_dataset_images_tp1[start[i]:end[i]], \
                                                   self.datasets.train_dataset_joints[start[i]:end[i]], \
                                                   self.datasets.train_dataset_cmd[start[i]:end[i]], \
                                                   self.datasets.train_dataset_optical_flow[start[i]:end[i]],\
                                                   save_gif=save_gif)

    def get_fusion_weights(self):
        #return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])
        return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])

    def plot_predictions(self, filename, images_t, images_tp1, joints, commands, opt_flow, save_gif=False):
        predictions_all_outputs = self.model.predict([images_t, joints, commands])
        if self.parameters.get('model_auxiliary'):
            predictions = predictions_all_outputs[0]
        else:
            predictions = predictions_all_outputs

        w_v = np.ones(shape=[len(images_t),256])*0.8
        w_j = np.ones(shape=[len(images_t),256])*0.1
        w_m = np.ones(shape=[len(images_t),256])*0.1
        pred_pre_fusion = self.model_pre_fusion([images_t, joints, commands])
        #print(pred_pre_fusion[0].shape)
        pred_custom_fusion_allvision = self.model_custom_fusion.predict([pred_pre_fusion[0], w_v,
                                                                 pred_pre_fusion[1], w_j,
                                                                 pred_pre_fusion[2], w_m])

        #print ('plotpred shape all ', np.asarray(predictions_all_outputs).shape)
        #print ('plotpred shape ', np.asarray(predictions).shape)
        # get activations of the fusion weights layer
        fusion_weights = self.model_fusion_weights.predict([images_t, \
                                                            joints, \
                                                            commands])
        bar_label = ['v', 'p', 'm']
        fig = plt.figure(figsize=(6, 10))
        for i in range(self.parameters.get('plots_predict_size')):
            # display original
            ax1 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 1)
            plt.imshow(images_t[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')), cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_ylabel('img(t)', rotation=0)
            if i != 0:
                ax1.get_yaxis().set_visible(False)

            ax2 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + self.parameters.get('plots_predict_size') + 1)
            plt.imshow(images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.set_ylabel('img(t+1)', rotation=0)
            if i != 0:
                ax2.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_imgp1_'+str(i)+ '.png', bbox_inches=extent_2)

            ax3 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 2 * (self.parameters.get('plots_predict_size')) + 1)
            opt_unnorm = deepcopy(opt_flow[i])
            if self.parameters.get('opt_flow_only_magnitude'):
                opt_unnorm = opt_unnorm * self.parameters.get('opt_flow_max_value')
            else:
                opt_unnorm = opt_unnorm[...,0] * self.parameters.get('opt_flow_max_value')
            plt.imshow(opt_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            ax3.set_ylabel('true OF', rotation=0)
            if i != 0:
                ax3.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_trueOF_'+str(i)+ '.png', bbox_inches=extent_3)

            ax4 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 3 * (self.parameters.get('plots_predict_size')) + 1)
            pred_unnorm = deepcopy(predictions[i])
            #print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
            if self.parameters.get('opt_flow_only_magnitude'):
                pred_unnorm = pred_unnorm * self.parameters.get('opt_flow_max_value')
            else:
                pred_unnorm = pred_unnorm[...,0] * self.parameters.get('opt_flow_max_value')
            plt.imshow(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax4.get_xaxis().set_visible(False)
            ax4.get_yaxis().set_visible(False)
            ax4.set_ylabel('pred.OF', rotation=0)
            if i != 0:
                ax4.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_predOF_'+str(i)+ '.png', bbox_inches=extent_4)

            ax5 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 4 * (self.parameters.get('plots_predict_size')) + 1)
            ax5.set_ylim(0, 1)
            plt.bar(bar_label, fusion_weights[i])
            ax5.set_ylabel('fus. w.', rotation=0)
            if i != 0:
                ax5.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_5 = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_fw_'+str(i)+ '.png', bbox_inches=extent_5)

            ax6 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 5 * (self.parameters.get('plots_predict_size')) + 1)
            attenuated_image_tp1=sensory_attenuation(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                                images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                                self.datasets.background_image)
            plt.imshow(attenuated_image_tp1, cmap='gray')
            ax6.get_xaxis().set_visible(False)
            ax6.get_yaxis().set_visible(False)
            ax6.set_ylabel('att.(t+1)', rotation=0)
            if i != 0:
                ax6.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_6 = ax6.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_attenuated_'+str(i)+ '.png', bbox_inches=extent_6)

            ax7 = plt.subplot(10, self.parameters.get('plots_predict_size'), i + 6 * (self.parameters.get('plots_predict_size')) + 1)
            ax7.set_ylim(0, 1)
            plt.bar(bar_label, [w_v[0,0], w_j[0,0], w_m[0,0]])
            ax7.set_ylabel('custom w', rotation=0)
            if i != 0:
                ax7.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_7 = ax7.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_custom_fw_'+str(i)+ '.png', bbox_inches=extent_7)

            ax8 = plt.subplot(10, self.parameters.get('plots_predict_size'),
                              i + 7 * (self.parameters.get('plots_predict_size')) + 1)
            predcustom_unnorm = deepcopy(pred_custom_fusion_allvision[i])
            #print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
            if self.parameters.get('opt_flow_only_magnitude'):
                predcustom_unnorm = predcustom_unnorm * self.parameters.get('opt_flow_max_value')
            else:
                predcustom_unnorm = predcustom_unnorm[...,0] * self.parameters.get('opt_flow_max_value')
            plt.imshow(predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax8.get_xaxis().set_visible(False)
            ax8.get_yaxis().set_visible(False)
            ax8.set_ylabel('custom prOF', rotation=0)
            if i != 0:
                ax8.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_8 = ax8.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_custom_predOF_'+str(i)+ '.png', bbox_inches=extent_8)

            ax9 = plt.subplot(10, self.parameters.get('plots_predict_size'),
                              i + 8 * (self.parameters.get('plots_predict_size')) + 1)
            attenuated_custom = sensory_attenuation(
                predcustom_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                self.datasets.background_image)
            plt.imshow(attenuated_custom, cmap='gray')
            ax9.get_xaxis().set_visible(False)
            ax9.get_yaxis().set_visible(False)
            ax9.set_ylabel('att.custom', rotation=0)
            if i != 0:
                ax9.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_9 = ax9.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif') + filename + '_attenuated_custom_' + str(i) + '.png',
                            bbox_inches=extent_9)

        plt.savefig(self.parameters.get('directory_plots') + filename + '.png')

        return fusion_weights

    def plot_predictions_test_dataset(self, epoch, logs):
        print('Callback: saving predicted images')
        fusion_weights = self.plot_predictions('predictions_epoch_' + str(epoch),\
                                               self.datasets.test_dataset_images_t[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_images_tp1[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_joints[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_cmd[0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.test_dataset_optical_flow[0:self.parameters.get('plots_predict_size')])

        np.savetxt(self.parameters.get('directory_plots') + "fusion_weights_" + str(epoch) + ".txt", fusion_weights,
                   fmt="%s")

