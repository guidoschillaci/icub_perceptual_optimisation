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

# the activation function of the output layer of the model
def activation_opt_flow(x):
    # apply activation
    x0 = K.relu(x[..., 0])  # magnitude is normalised between 0 and 1
    x1 = K.tanh(x[..., 1])  # cos(alpha) can have values between -1 and 1
    x2 = K.tanh(x[..., 2])  # sin(alpha) can have values between -1 and 1
    #out = [x0,x1,x2]
    return tf.stack((x0,x1,x2),axis=-1)


class Split(tf.keras.layers.Layer):
    def __init__(self):
        super(Split, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, input, **kwargs):
        return tf.split(input, 3, axis=1)


class MyCallback(Callback):
    def __init__(self, param, datasets, model):
        self.parameters =param
        self.datasets = datasets
        if self.parameters.get('model_custom_training_loop'):
            self.model = model
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

        if self.parameters.get('make_plots'):
        #    # plot also sequences of predictions
            self.plot_train_sequences(save_gif=True)
        #    self.plot_predictions_test_dataset(epoch, logs, predict_size=self.parameters.get('plots_predict_size'))

    def save_plots(self):
        pd.DataFrame.from_dict(self.history).to_csv(self.parameters.get('directory_results') +'history.csv', index=False)
        # history dictioary
        history_keys = list(self.history.keys())
        #print('keras history keys ', history_keys)

        # summarize history for loss
        fig = plt.figure(figsize=(10, 10))
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
                                                   self.datasets.dataset_images_t[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_images_tp1[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_joints[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_cmd[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_optical_flow[self.datasets.train_indexes][start[i]:end[i]],\
                                                   save_gif=save_gif)

    def get_fusion_weights(self):
        #return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])
        return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])

    def plot_predictions(self, filename, images_t, images_tp1, joints, commands, opt_flow, save_gif=False):
        predictions_all_outputs = self.model.predict([images_t, joints, commands])
        predictions = predictions_all_outputs[0]
        #print ('plotpred shape all ', np.asarray(predictions_all_outputs).shape)
        #print ('plotpred shape ', np.asarray(predictions).shape)
        # get activations of the fusion weights layer
        fusion_weights = self.model_fusion_weights.predict([images_t, \
                                                            joints, \
                                                            commands])
        bar_label = ['v', 'p', 'm']
        fig = plt.figure(figsize=(12, 4))
        for i in range(self.parameters.get('plots_predict_size')):
            # display original
            ax1 = plt.subplot(5, self.parameters.get('plots_predict_size'), i + 1)
            plt.imshow(images_t[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')), cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.set_ylabel('img(t)', rotation=0)
            if i != 0:
                ax1.get_yaxis().set_visible(False)

            ax2 = plt.subplot(5, self.parameters.get('plots_predict_size'), i + self.parameters.get('plots_predict_size') + 1)
            plt.imshow(images_tp1[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax2.get_xaxis().set_visible(False)
            ax2.set_ylabel('img(t+1)', rotation=0)
            if i != 0:
                ax2.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_imgp1_'+str(i)+ '.png', bbox_inches=extent_2)

            ax3 = plt.subplot(5, self.parameters.get('plots_predict_size'), i + 2 * (self.parameters.get('plots_predict_size')) + 1)
            opt_unnorm = deepcopy(opt_flow[i])
            if self.parameters.get('opt_flow_only_magnitude'):
                opt_unnorm = opt_unnorm * self.parameters.get('opt_flow_max_value')
            else:
                opt_unnorm = opt_unnorm[...,0] * self.parameters.get('opt_flow_max_value')
            plt.imshow(opt_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax3.get_xaxis().set_visible(False)
            ax3.set_ylabel('true OF', rotation=0)
            if i != 0:
                ax3.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_3 = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_trueOF_'+str(i)+ '.png', bbox_inches=extent_3)

            ax4 = plt.subplot(5, self.parameters.get('plots_predict_size'), i + 3 * (self.parameters.get('plots_predict_size')) + 1)
            pred_unnorm = deepcopy(predictions[i])

            #print('pred_unnorm shape ', np.asarray(pred_unnorm).shape)
            if self.parameters.get('opt_flow_only_magnitude'):
                pred_unnorm = pred_unnorm * self.parameters.get('opt_flow_max_value')
            else:
                pred_unnorm = pred_unnorm[...,0] * self.parameters.get('opt_flow_max_value')
            plt.imshow(pred_unnorm.reshape(self.parameters.get('image_size'), self.parameters.get('image_size')),
                       cmap='gray')
            ax4.get_xaxis().set_visible(False)
            ax4.set_ylabel('pred.OF', rotation=0)
            if i != 0:
                ax4.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_4 = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_predOF_'+str(i)+ '.png', bbox_inches=extent_4)

            ax5 = plt.subplot(5, self.parameters.get('plots_predict_size'), i + 4 * (self.parameters.get('plots_predict_size')) + 1)
            ax5.set_ylim(0, 1)
            plt.bar(bar_label, fusion_weights[i])
            ax5.set_ylabel('fus. w.', rotation=0)
            if i != 0:
                ax5.get_yaxis().set_visible(False)
            if save_gif:
                # Save just the portion _inside_ the second axis's boundaries
                extent_5 = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(self.parameters.get('directory_plots_gif')+ filename +'_fw_'+str(i)+ '.png', bbox_inches=extent_5)
        plt.savefig(self.parameters.get('directory_plots') + filename + '.png')

        return fusion_weights

    def plot_predictions_test_dataset(self, epoch, logs):
        print('Callback: saving predicted images')
        fusion_weights = self.plot_predictions('predictions_epoch_' + str(epoch),\
                                               self.datasets.dataset_images_t[self.datasets.test_indexes][0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.dataset_images_tp1[self.datasets.test_indexes][0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.dataset_joints[self.datasets.test_indexes][0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.dataset_cmd[self.datasets.test_indexes][0:self.parameters.get('plots_predict_size')], \
                                               self.datasets.dataset_optical_flow[self.datasets.test_indexes][0:self.parameters.get('plots_predict_size')])

        np.savetxt(self.parameters.get('directory_plots') + "fusion_weights_" + str(epoch) + ".txt", fusion_weights,
                   fmt="%s")
