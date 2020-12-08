from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd

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
    def __init__(self, param, datasets):
        self.parameters =param
        self.datasets = datasets

    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        print('callback train begin')
        #print('train size ', str(len(self.datasets.dataset_images_t[[self.datasets.train_indexes]])))
        #print('teest size ', str(len(self.datasets.dataset_images_t[[self.datasets.test_indexes]])))
        # sub model with fusion weights output
        self.model_fusion_weights = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(name='fusion_weights').output)
        if self.parameters.get('make_plots'):
            self.plot_predictions_test_dataset(-1, logs)
            # plot also sequences of predictions
            self.plot_train_sequences(predict_size=self.parameters.get('plots_predict_size'))

    def on_batch_end (self, batch, logs={}):
        pass

    def on_train_end(self, logs=None):
        print('callback train end')
        if self.parameters.get('make_plots'):
            # plot also sequences of predictions
            self.plot_train_sequences(predict_size=self.parameters.get('plots_predict_size'),\
                                      save_gif=True)

        self.save_plots()

    def on_epoch_end(self, epoch, logs=None):
        print('callback epoch end')

        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('acc'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_acc'))
        #if self.parameters.get('make_plots'):
        #    # plot also sequences of predictions
        #    self.plot_train_sequences(predict_size=self.parameters.get('plots_predict_size'))
        #    self.plot_predictions_test_dataset(epoch, logs, predict_size=self.parameters.get('plots_predict_size'))

    def save_plots(self):
        pd.DataFrame.from_dict(self.history).to_csv(self.parameters.get('directory_results') +'history.csv', index=False)
        # history dictioary
        history_keys = self.history.keys()
        print('keras history keys ', history_keys)

        # summarize history for loss
        fig = plt.figure(figsize=(10, 10))
        plt.title('model history')
        plt.ylabel('value')
        plt.xlabel('epoch')
        for i in range(len(history_keys)):

            plt.plot(self.history[history_keys[i]], label=history_keys[i])
            np.savetxt(self.parameters.get('directory_plots') + history_keys[i]+ '.txt', self.history[history_keys[i]],fmt="%s")

        plt.legend(history_keys, loc='upper left')
        plt.savefig(self.parameters.get('directory_plots') + 'history.png')
        #plt.show()

    def plot_train_sequences(self, predict_size=20, save_gif=False):
        start = [510, 700, 1300, 2500, 3600, 3780, 4570, 13900]
        end = list(np.asarray(start) + predict_size)
        for i in range(len(start)):
            print('plotting train from ' + str(start[i]) + ' to ' + str(end[i]))
            fusion_weights = self.plot_predictions('pred_sequence_train_' + str(start[i]) + '_' + str(end[i]), \
                                                   self.datasets.dataset_images_t[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_images_tp1[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_joints[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_cmd[self.datasets.train_indexes][start[i]:end[i]], \
                                                   self.datasets.dataset_optical_flow[self.datasets.train_indexes][start[i]:end[i]],\
                                                   predict_size = predict_size,\
                                                   save_gif=save_gif)

    def get_fusion_weights(self):
        #return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])
        return K.function([self.model.layers[0].input], [self.model.get_layer('fusion_weights').output])

    def plot_predictions(self, filename, images_t, images_tp1, joints, commands, opt_flow, predict_size=20, save_gif=False):
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
        for i in range(predict_size):
            # display original
            ax1 = plt.subplot(5, predict_size, i + 1)
            plt.imshow(images_t[i].reshape(self.parameters.get('image_size'), self.parameters.get('image_size')), cmap='gray')
            ax1.get_xaxis().set_visible(False)
            ax1.set_ylabel('img(t)', rotation=0)
            if i != 0:
                ax1.get_yaxis().set_visible(False)

            ax2 = plt.subplot(5, predict_size, i + predict_size + 1)
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

            ax3 = plt.subplot(5, predict_size, i + 2 * (predict_size) + 1)
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

            ax4 = plt.subplot(5, predict_size, i + 3 * (predict_size) + 1)
            pred_unnorm = deepcopy(predictions_all_outputs[i])

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

            ax5 = plt.subplot(5, predict_size, i + 4 * (predict_size) + 1)
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

    def plot_predictions_test_dataset(self, epoch, logs, predict_size=20):
        print('Callback: saving predicted images')
        fusion_weights = self.plot_predictions('predictions_epoch_' + str(epoch),\
                                               self.datasets.dataset_images_t[self.datasets.test_indexes][0:predict_size], \
                                               self.datasets.dataset_images_tp1[self.datasets.test_indexes][0:predict_size], \
                                               self.datasets.dataset_joints[self.datasets.test_indexes][0:predict_size], \
                                               self.datasets.dataset_cmd[self.datasets.test_indexes][0:predict_size], \
                                               self.datasets.dataset_optical_flow[self.datasets.test_indexes][0:predict_size])

        np.savetxt(self.parameters.get('directory_plots') + "fusion_weights_" + str(epoch) + ".txt", fusion_weights,
                   fmt="%s")
