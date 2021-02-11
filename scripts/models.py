import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPooling2D,UpSampling2D, Reshape, Concatenate, Add, Multiply, Softmax, ActivityRegularization, Layer
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.metrics import Mean
from tensorflow import keras as tfk
from utils import Split, MyCallback, activation_opt_flow
#from keract import get_activations, display_activations
from copy import deepcopy
import sys
import time

import numpy as np
import h5py
import dataset_loader
from tqdm import tqdm
import os
import tkinter
import matplotlib.pyplot as plt
import pandas as pd

#mae_metric = tfk.metrics.MeanSquaredError(name="mae")

class CustomModel(Model):
    def set_param(self, param):
        self.parameters=param
        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.val_loss_tracker = tfk.metrics.Mean(name="val_loss")

    def link_fusion_model(self, fusion_model):
        self.fusion_model = fusion_model

    def weight_loss(self, loss_aux_mod, w, fact):
        _shape = (self.parameters.get('image_size'), self.parameters.get('image_size'))
        # add dimension
        x = tf.expand_dims(w, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0]
        x = tf.tile(x, [1,_shape[1]])
        # add dimension
        x = tf.expand_dims(x, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0, image_shape_1]
        weight = tf.tile(x, [1, _shape[0], 1])
        alpha_weight = tf.math.scalar_mul(fact, tf.identity(weight))
        return loss_aux_mod * alpha_weight

        #fact_matrix = tf.math.scalar_mul(fact, tf.ones_like(w))
        ##sig_soft_loss_aux = tf.nn.softmax(tf.math.sigmoid(tf.math.exp(-tf.math.pow(loss_modality, 2))))
        #sig_soft_loss_aux = (tf.math.sigmoid(tf.math.exp(-tf.math.pow(loss_modality, 2))))
        #return fact_matrix * tf.math.pow((w - sig_soft_loss_aux), 2)

    @tf.function
    def loss_fn(self, y_true, y_pred, weights=[]):

        #print('size y_true', str(np.asarray(y_true).shape))
        #print('size y_pred', str(np.asarray(y_pred).shape))
        true_main_out = y_true[0]
        pred_main_out = y_pred[0]
        loss_main_out = tf.keras.losses.mean_squared_error(pred_main_out, true_main_out)
        if not self.parameters.get('model_auxiliary'):
            return loss_main_out
        else:
            true_aux_visual = y_true[1]
            true_aux_proprio = y_true[2]
            true_aux_motor = y_true[3]
            pred_aux_visual = y_pred[1]
            pred_aux_proprio = y_pred[2]
            pred_aux_motor = y_pred[3]

            #print('size y_true[1]', str(np.asarray(y_true[1]).shape))
            #print('size y_pred[1]', str(np.asarray(y_pred[1]).shape))
            #print('size y_true[2]', str(np.asarray(y_true[2]).shape))
            #print('size y_pred[2]', str(np.asarray(y_pred[2]).shape))
            #print('size y_true[3]', str(np.asarray(y_true[3]).shape))
            #print('size y_pred[3]', str(np.asarray(y_pred[3]).shape))

            alpha = self.parameters.get('model_sensor_fusion_alpha')  # 0.6 is good
            #beta = self.parameters.get('model_sensor_fusion_beta')  # 0.0
            #loss_aux_visual = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_visual), tf.squeeze(pred_aux_visual)))
            #loss_aux_proprio = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_proprio), tf.squeeze(pred_aux_proprio)))
            #loss_aux_motor = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_motor), tf.squeeze(pred_aux_motor)))
            loss_aux_visual = tf.keras.losses.mean_squared_error(true_aux_visual, pred_aux_visual)
            loss_aux_proprio = tf.keras.losses.mean_squared_error(true_aux_proprio, pred_aux_proprio)
            loss_aux_motor = tf.keras.losses.mean_squared_error(true_aux_motor, pred_aux_motor)

            #print('size loss_main_out', str(loss_main_out.numpy().shape))
            #print('size loss_aux_visual', str(loss_aux_visual.numpy().shape))
            #print('size loss_aux_proprio', str(loss_aux_proprio.numpy().shape))
            #print('size loss_aux_motor', str(loss_aux_motor.numpy().shape))
            #print('size weights', str(weights.numpy().shape))

            aux_loss_weighting_total = self.weight_loss(loss_aux_visual, weights[:,0], alpha) + \
                                       self.weight_loss(loss_aux_proprio, weights[:,1], alpha) + \
                                       self.weight_loss(loss_aux_motor, weights[:,2], alpha)
            #aux_loss_weighting_total = tf.reduce_mean(self.weight_loss(loss_aux_visual, weights[:, 0], alpha)) + \
            #                           tf.reduce_mean(self.weight_loss(loss_aux_proprio, weights[:, 1], alpha)) + \
            #                           tf.reduce_mean(self.weight_loss(loss_aux_motor, weights[:, 2], alpha))

            #fus_weight_regul_total = tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_visual, weights[:,0], beta)) + \
            #                         tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_proprio,weights[:,1], beta)) + \
            #                         tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_motor,  weights[:,2], beta))
            #print('shape fus_weight_regul_total', str(fus_weight_regul_total.numpy().shape))

            #reg_fact = [tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_visual, weights[:,0], beta)), \
            #            tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_proprio,weights[:,1], beta)), \
            #            tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_motor,  weights[:,2], beta))]
            #print('shape reg_fact', str(np.asarray(reg_fact).shape))
            self.get_layer('fusion_activity_regularizer_layer').set_regularization_factors(weights)
            print ('layer reg ', self.get_layer('fusion_activity_regularizer_layer').reg_fact)
            return loss_main_out + aux_loss_weighting_total# + fus_weight_regul_total

    def train_step(self, data):
        if self.parameters.get('model_auxiliary'):
            (in_img, in_j, in_cmd), (out_of, out_aof1, out_aof2, out_aof3) = data
            weights_predictions = self.fusion_model((in_img, in_j, in_cmd))
            with tf.GradientTape() as tape:
                # forward pass
                predictions = self((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = self.loss_fn((out_of, out_aof1, out_aof2, out_aof3), \
                                          predictions, \
                                          weights=weights_predictions)
                # Add any extra losses created during the forward pass.
                loss_value += sum(self.losses)

            grads = tape.gradient(loss_value, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(loss_value)
                # self.train_callback.on_batch_end(batch=-1, logs=self.logs)
            return {"loss": self.loss_tracker.result()}
        else:  # simple model
            (in_img, in_j, in_cmd), out_of = data
            with tf.GradientTape() as tape:
                # forward pass
                predictions = self((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = tf.keras.losses.mean_squared_error(out_of, predictions)
                # Add any extra losses created during the forward pass.
                loss_value += sum(self.losses)
            # compute gradients
            grads = tape.gradient(loss_value, self.trainable_weights)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(loss_value)
            # self.train_callback.on_batch_end(batch=-1, logs=self.logs)
            return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        if self.parameters.get('model_auxiliary'):
            (in_img, in_j, in_cmd), (out_of, out_aof1, out_aof2, out_aof3) = data
            weights_predictions = self.fusion_model((in_img, in_j, in_cmd))
            predictions = self((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
            # Compute the loss value for this minibatch.
            val_loss_value = self.loss_fn((out_of, out_aof1, out_aof2, out_aof3), \
                                                   predictions, \
                                                   weights=weights_predictions)
            # Add any extra losses created during the forward pass.
            val_loss_value += sum(self.losses)
            self.val_loss_tracker.update_state(val_loss_value)
            return {"val_loss": self.val_loss_tracker.result()}
        else: # simple model
            (in_img, in_j, in_cmd), out_of  = data
            predictions = self((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
            val_loss_value = tf.keras.losses.mean_squared_error(out_of, predictions)
            # Add any extra losses created during the forward pass.
            val_loss_value += sum(self.losses)
            self.val_loss_tracker.update_state(val_loss_value)
            return {"val_loss": self.val_loss_tracker.result()}
    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

class FusionActivityRegularizationLayer(Layer):
    def __init__(self, param, **kwargs):
        super(FusionActivityRegularizationLayer, self).__init__(**kwargs)
        self.parameters = param
        # self.reg_fact = [0.33, 0.33, 0.33]
        self.reg_fact = tf.fill([self.parameters.get('model_batch_size'), 3], 0.33)
        self.beta = self.parameters.get('model_sensor_fusion_beta')

    def set_regularization_factors(self, reg_fact):
        self.reg_fact = reg_fact

    def fusion_weights_regulariser(self, input, w, fact):
        #_shape = (self.parameters.get('image_size'), self.parameters.get('image_size'))
        # add dimension
        x = tf.expand_dims(w, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0]
        #x = tf.tile(x, [1, _shape[1]])
        # add dimension
        #x = tf.expand_dims(x, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0, image_shape_1]
        weight = tf.tile(x, [1, 3, 1])
        fact_matrix = tf.math.scalar_mul(fact, tf.ones_like(weight))
        sig_soft_loss_aux = tf.nn.softmax(tf.math.sigmoid(tf.math.exp(-tf.math.pow(input, 2))))
        #sig_soft_loss_aux = tf.math.sigmoid(tf.math.exp(-tf.math.pow(loss_modality, 2)))
        return fact_matrix * tf.math.pow((weight - sig_soft_loss_aux), 2)

    def call(self, inputs):
        #print('reg fact ', self.reg_fact)
        Z = tf.reduce_mean(self.fusion_weights_regulariser(inputs, self.reg_fact[:, 0], self.beta)) + \
            tf.reduce_mean(self.fusion_weights_regulariser(inputs, self.reg_fact[:, 1], self.beta)) + \
            tf.reduce_mean(self.fusion_weights_regulariser(inputs, self.reg_fact[:, 2], self.beta))
        #Z = inputs[0]*self.reg_fact[0] + inputs[1]*self.reg_fact[1] + inputs[2]*self.reg_fact[2]
        self.add_loss(Z)
        return inputs  # Pass-through layer.

class Models:
    def __init__(self, param):
        print('creating models')
        self.parameters = param

    def read_data(self):
        self.datasets = dataset_loader.DatasetLoader(self.parameters)
        self.datasets.load_datasets()

    def make_model(self):
        # inspired by NetGate model, Patel et al., (2017) "Sensor Modality Fusion with CNNs
        # for UGV Autonomous Driving in Indoor Environments", IROS

        # visual branch
        input_visual = Input(shape=(self.parameters.get('image_size'), self.parameters.get('image_size'), self.parameters.get('image_channels')))
        visual_layer_1 = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                   activation='relu', padding='same')
        visual_layer_2 = Dropout(0.4)
        visual_layer_3 = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                   activation='relu', padding='same')
        visual_layer_4 = Dropout(0.4)
        visual_layer_5 = MaxPooling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')), \
                         padding='same')
        visual_layer_6 = Dropout(0.4)
        visual_layer_7 = Conv2D(16, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                   activation='relu', padding='same')
        visual_layer_8 = Dropout(0.4)
        visual_layer_9 = Conv2D(16, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                   activation='relu', padding='same')
        visual_layer_10 = Dropout(0.4)
        visual_layer_11 = MaxPooling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')), \
                         padding='same')
        visual_layer_12 = Dropout(0.4)
        visual_layer_13 = Conv2D(32, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                   activation='relu', padding='same')
        visual_layer_14 = Dropout(0.4)
        visual_layer_15 = MaxPooling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')), \
                         padding='same')
        visual_layer_16 = Flatten()
        visual_layer_17 = Dense(256, activation='relu')
        ## link layers of the visual branch of the main model
        out_visual_main = visual_layer_17 ( \
            visual_layer_16 ( \
            visual_layer_15( \
            visual_layer_14( \
            visual_layer_13( \
            visual_layer_12( \
            visual_layer_11( \
            visual_layer_10( \
            visual_layer_9( \
            visual_layer_8( \
            visual_layer_7( \
            visual_layer_6( \
            visual_layer_5( \
            visual_layer_4( \
            visual_layer_3( \
            visual_layer_2( \
            visual_layer_1( \
            input_visual) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

        # proprioceptive (joint encoders) branch
        input_proprioceptive = Input( shape=(len(self.datasets.dataset_joints[0]),) )
        proprioceptive_layer_1 = Dense(2 * len(self.datasets.dataset_joints[0]), activation='relu')
        proprioceptive_layer_2 = Dropout(0.4)
        proprioceptive_layer_3 = Dense(4 * len(self.datasets.dataset_joints[0]), activation='relu')
        proprioceptive_layer_4 = Dropout(0.4)
        proprioceptive_layer_5 = Dense(8 * len(self.datasets.dataset_joints[0]), activation='relu')
        proprioceptive_layer_6 = Dropout(0.4)
        proprioceptive_layer_7 = Dense(256, activation='relu')
        ## link layers of the proprioceptive branch of the main model
        out_proprioceptive_main = proprioceptive_layer_7 ( \
            proprioceptive_layer_6( \
            proprioceptive_layer_5( \
            proprioceptive_layer_4( \
            proprioceptive_layer_3( \
            proprioceptive_layer_2( \
            proprioceptive_layer_1( \
            input_proprioceptive) ) ) ) ) ) )

        # motor commands branch
        input_motor = Input( shape=(len(self.datasets.dataset_cmd[0]),) )
        motor_layer_1 = Dense(2 * len(self.datasets.dataset_cmd[0]), activation='relu')
        motor_layer_2 = Dropout(0.4)
        motor_layer_3 = Dense(4 * len(self.datasets.dataset_cmd[0]), activation='relu')
        motor_layer_4 = Dropout(0.4)
        motor_layer_5 = Dense(8 * len(self.datasets.dataset_cmd[0]), activation='relu')
        motor_layer_6 = Dropout(0.4)
        motor_layer_7 = Dense(256, activation='relu')
        ## link layers of the motor branch of the main model
        out_motor_main = motor_layer_7 ( \
            motor_layer_6 ( \
            motor_layer_5 ( \
            motor_layer_4 ( \
            motor_layer_3 ( \
            motor_layer_2 ( \
            motor_layer_1 ( \
            input_motor) ) ) ) ) ) )

        concatenated = Concatenate()([out_visual_main, out_proprioceptive_main, out_motor_main])
        x = Dense(16)(concatenated)
        x = Dense(3, activation='sigmoid')(x)
        #x = Dense(3, activation='relu')(x)
        #x = ActivityRegularization(l1=0.01, name='act_regularizer')(x) #
        x = FusionActivityRegularizationLayer(param=self.parameters, name='fusion_activity_regularizer_layer')(x)  #
        fusion_weight_layer = Softmax(axis=-1, name='fusion_weights')(x) # makes weights sum up to 1
        # get fusion weights
        fusion_weight_visual, fusion_weight_proprio, fusion_weight_motor = Split()(fusion_weight_layer)

        weighted_visual = Multiply(name='weighted_visual')
        weighted_proprio = Multiply(name='weighted_proprio')
        weighted_motor = Multiply(name='weighted_motor')
        addition = Add()
        final_1 = Dense(256, activation='relu')
        final_2 = Reshape(target_shape=(16, 16, 1))
        final_3 = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')),
                         activation='relu', \
                         padding='same')
        final_4 = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))
        final_5 = Conv2D(1, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                                    activation='relu', \
                                    padding='same', name='main_output')

        # link them
        out_main_model = final_5(final_4(final_3(final_2(final_1(addition([weighted_visual([out_visual_main, fusion_weight_visual]),
                                                                          weighted_proprio([out_proprioceptive_main, fusion_weight_proprio]),
                                                                          weighted_motor([out_motor_main, fusion_weight_motor])]
                                                                          ) ) ) ) ) )


        # weighted_visual = Multiply(name='weighted_visual')([out_visual_main, fusion_weight_visual])
        # weighted_proprio = Multiply(name='weighted_proprio')([out_proprioceptive_main, fusion_weight_proprio])
        # weighted_motor = Multiply(name='weighted_motor')([out_motor_main, fusion_weight_motor])
        # addition = Add()([weighted_visual, weighted_proprio, weighted_motor])
        #final_1 = Dense(256, activation='relu')(addition)
        #final_2 = Reshape(target_shape=(16, 16, 1))(final_1)
        #final_3 = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), activation='relu', \
        #           padding='same')(final_2)
        #final_4 = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(final_3)
        #if self.parameters.get('image_size')==64:
        #    final_5 = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), activation='relu', \
        #               padding='same')(final_4)
        #    final_6 = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(final_5)
        #    if self.parameters.get('opt_flow_only_magnitude'):
        #        out_main_model = Conv2D(1, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
        #                       activation='relu', \
        #                       padding='same', name='main_output')(final_6)
        #    else:
        #        out_main_model = Conv2D(3, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
        #               activation=activation_opt_flow, \
        #               padding='same', name='main_output')(final_6)
        #else:
        #    if self.parameters.get('opt_flow_only_magnitude'):
        #        out_main_model = Conv2D(1, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
        #                       activation='relu', \
        #                       padding='same', name='main_output')(final_4)
        #    else:
        #        out_main_model = Conv2D(3, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
        #               activation=activation_opt_flow, \
        #               padding='same', name='main_output')(final_4)

        if self.parameters.get('model_auxiliary'):

            ##########
            # Auxiliary model for weight regularisation
            # inspired by https: // arxiv.org / pdf / 1901.10610.pdf
            ## visual branch
            aux_visual_layer_1 = visual_layer_17 ( \
                visual_layer_16 ( \
                visual_layer_15( \
                visual_layer_14( \
                visual_layer_13( \
                visual_layer_12( \
                visual_layer_11( \
                visual_layer_10( \
                visual_layer_9( \
                visual_layer_8( \
                visual_layer_7( \
                visual_layer_6( \
                visual_layer_5( \
                visual_layer_4( \
                visual_layer_3( \
                visual_layer_2( \
                visual_layer_1( \
                input_visual) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )
            x = Dense(256, activation='relu')(aux_visual_layer_1)
            x = Reshape(target_shape=(16, 16, 1))(x)
            x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), activation='relu', \
                       padding='same')(x)
            x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('image_size') == 64:
                x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')),
                           activation='relu', \
                           padding='same')(x)
                x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('opt_flow_only_magnitude'):
                out_visual_aux_model = Conv2D(1, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                                   activation='relu', \
                                   padding='same',
                                   name='aux_visual_output')(x)
            else:
                out_visual_aux_model = Conv2D(3, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                           activation=activation_opt_flow, \
                           padding='same',\
                           name='aux_visual_output')(x)

            ## proprioceptive branch
            aux_proprioceptive_layer_1 = proprioceptive_layer_7 ( \
                proprioceptive_layer_6 ( \
                proprioceptive_layer_5 ( \
                proprioceptive_layer_4 ( \
                proprioceptive_layer_3 ( \
                proprioceptive_layer_2 ( \
                proprioceptive_layer_1 ( \
                input_proprioceptive) ) ) ) ) ) )
            x = Dense(256, activation='relu')(aux_proprioceptive_layer_1)
            x = Reshape(target_shape=(16, 16, 1))(x)
            x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), activation='relu', \
                       padding='same')(x)
            x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('image_size') == 64:
                x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')),
                           activation='relu', \
                           padding='same')(x)
                x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('opt_flow_only_magnitude'):
                out_proprio_aux_model = Conv2D(1, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                                   activation='relu', \
                                   padding='same', \
                                name='aux_proprio_output')(x)
            else:
                out_proprio_aux_model = Conv2D(3, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                           activation=activation_opt_flow, \
                           padding='same',\
                           name='aux_proprio_output')(x)

            # motor branch
            aux_motor_layer_1 = motor_layer_7 ( \
                motor_layer_6 ( \
                motor_layer_5 ( \
                motor_layer_4 ( \
                motor_layer_3 ( \
                motor_layer_2 ( \
                motor_layer_1 ( \
                input_motor) ) ) ) ) ) )
            x = Dense(256, activation='relu')(aux_motor_layer_1)
            x = Reshape(target_shape=(16, 16, 1))(x)
            x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')),
                       activation='relu', \
                       padding='same')(x)
            x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('image_size') == 64:
                x = Conv2D(8, (self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')),
                           activation='relu', \
                           padding='same')(x)
                x = UpSampling2D((self.parameters.get('model_max_pool_size'), self.parameters.get('model_max_pool_size')))(x)
            if self.parameters.get('opt_flow_only_magnitude'):
                out_motor_aux_model = Conv2D(1, (
                self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                                               activation='relu', \
                                               padding='same', \
                                               name='aux_motor_output')(x)
            else:
                out_motor_aux_model = Conv2D(3, (
                self.parameters.get('model_conv_size'), self.parameters.get('model_conv_size')), \
                                               activation=activation_opt_flow, \
                                               padding='same',\
                                               name='aux_motor_output')(x)

            # define the model
            self.model = CustomModel(inputs=[input_visual, input_proprioceptive, input_motor],
                                     outputs=[out_main_model, out_visual_aux_model, out_proprio_aux_model, out_motor_aux_model] )
            self.model.compile(optimizer='adam')
            self.model.set_param(self.parameters)
            # create a new model sharing the parameters of the main one, to be used only for predicting modality weights
            self.fusion_weights_model = Model(inputs=self.model.input,
                                              outputs=self.model.get_layer(name='fusion_weights').output)
            # link this model
            self.model.link_fusion_model(self.fusion_weights_model)

            # we need an additional model (sharing the parameters) for manually setting the fusion weights
            #### model allowing manually setting the fusion weights
            # first: pre_fusion model give pre-fusion outputs for each modality
            self.model_pre_fusion = Model(inputs=self.model.input,outputs=[out_visual_main,
                                                                           out_proprioceptive_main,
                                                                           out_motor_main])
            # second model takes as inputs these pre-fusion outputs and the custom fusion weights
            self.custom_weight_visual_inp = Input(shape=(1,))
            self.custom_weight_proprio_inp = Input(shape=(1,))
            self.custom_weight_motor_inp = Input(shape=(1,))
            self.custom_visual_inp = Input(shape=(256,))
            self.custom_proprio_inp = Input(shape=(256,))
            self.custom_motor_inp = Input(shape=(256,))

            # link them
            self.out_custom_weight_model = final_5(
                final_4(final_3(
                final_2(final_1(addition([weighted_visual([self.custom_visual_inp, self.custom_weight_visual_inp]),
                                          weighted_proprio([self.custom_proprio_inp, self.custom_weight_proprio_inp]),
                                          weighted_motor([self.custom_motor_inp, self.custom_weight_motor_inp])]
                                                         ))))))
            self.model_custom_fusion = Model(inputs=[self.custom_visual_inp, self.custom_weight_visual_inp,
                                              self.custom_proprio_inp, self.custom_weight_proprio_inp,
                                              self.custom_motor_inp, self.custom_weight_motor_inp],
                                            outputs=self.out_custom_weight_model)

        #self.model_custom_fusion = Model(inputs=(self.model.input, self.model.get_layer(name='fusion_weights'),
            #                                  outputs=self.model.get_layer(name='fusion_weights').output)
            #self.model.set_param(self.parameters)

        ##########
        else:
            # without auxiliary model
            # define the model
            self.model = CustomModel(inputs=[input_visual, input_proprioceptive, input_motor],
                                     outputs=out_main_model)
            self.model.set_param(self.parameters)
            #if not self.parameters.get('model_custom_training_loop'):
            self.model.compile(optimizer='adam',loss='mean_squared_error')
            #else:
            #    self.optimiser = Adam()
            #    self.train_callback = MyCallback(self.parameters, self.datasets, self.model)
            #    self.logs = {}

        if self.parameters.get('verbosity_level') >0:
            self.model.summary()

    def train_model(self):
        #print('Custom training loop? ', str(self.parameters.get('model_custom_training_loop')))
        #if self.parameters.get('model_custom_training_loop'):
        #    self.custom_training_loop()
        #else:
        #self.keras_training_loop()
        print('starting training the model with keras fit function')
        self.myCallback = MyCallback(self.parameters, self.datasets, self.model, self.model_pre_fusion, self.model_custom_fusion)
        if self.parameters.get('model_auxiliary'):
            fusion_weights_train = self.fusion_weights_model.predict( \
                [self.datasets.train_dataset_images_t, \
                 self.datasets.train_dataset_joints, \
                 self.datasets.train_dataset_cmd])

            fusion_weights_test = self.fusion_weights_model.predict( \
                [self.datasets.test_dataset_images_t, \
                 self.datasets.test_dataset_joints, \
                 self.datasets.test_dataset_cmd])

            self.history = self.model.fit([self.datasets.train_dataset_images_t, \
                                           self.datasets.train_dataset_joints, \
                                           self.datasets.train_dataset_cmd], \
                                          [self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow], \
                                          epochs=self.parameters.get('model_epochs'), \
                                          batch_size=self.parameters.get('model_batch_size'), \
                                          validation_data=([self.datasets.test_dataset_images_t, \
                                                            self.datasets.test_dataset_joints, \
                                                            self.datasets.test_dataset_cmd], \
                                                           [self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow]), \
                                          shuffle=True, \
                                          callbacks=[self.myCallback], \
                                          verbose=1)
        else:
            self.history = self.model.fit([self.datasets.train_dataset_images_t, \
                                           self.datasets.train_dataset_joints, \
                                           self.datasets.train_dataset_cmd], \
                                          self.datasets.train_dataset_optical_flow, \
                                          epochs=self.parameters.get('model_epochs'), \
                                          batch_size=self.parameters.get('model_batch_size'), \
                                          validation_data=([self.datasets.test_dataset_images_t, \
                                                            self.datasets.test_dataset_joints, \
                                                            self.datasets.test_dataset_cmd], \
                                                           self.datasets.test_dataset_optical_flow), \
                                          shuffle=True, \
                                          callbacks=[self.myCallback], \
                                          verbose=1)
        print('training done')


    def load_model(self):
        model_filename = self.parameters.get('directory_models') + self.parameters.get('model_filename')

        # if model file already exists (i.e. it has been already trained):
        if os.path.isfile(model_filename):
            # load mode
            self.model = load_model(model_filename) # keras.load_model function
            print('Loaded pre-trained network named: ', model_filename)

    def plot_model(self):
        print('saving plot of the model...')
        # model plot
        model_plt_file = self.parameters.get('directory_plots') + self.parameters.get('model_plot_filename')
        tf.keras.utils.plot_model(self.model, to_file=model_plt_file, show_shapes=True)

    def save_model(self):
        self.model.save(self.parameters.get('directory_models') + self.parameters.get('model_filename'), overwrite=True)
        self.plot_model()
        print('model saved')

    def save_plots(self):
        pd.DataFrame.from_dict(self.myCallback.history).to_csv(self.parameters.get('directory_results') +'history.csv', index=False)

        history_keys = list(self.myCallback.history.keys())
        print ('hisotry keys ', history_keys)

        # summarize history for loss
        fig = plt.figure(figsize=(10, 12))
        plt.title('model history')
        plt.ylabel('value')
        plt.xlabel('epoch')
        for i in range(len(history_keys)):
            #if (history_keys[i] == 'loss') or (history_keys[i]=='val_loss'):
            plt.plot(self.myCallback.history[history_keys[i]], label=history_keys[i])
            np.savetxt(self.parameters.get('directory_plots') + history_keys[i]+ '.txt', self.myCallback.history[history_keys[i]],fmt="%s")
        plt.legend(history_keys, loc='upper left')
        plt.savefig(self.parameters.get('directory_plots') + 'history.png')

    '''
    #@tf.function
    def custom_training_loop(self):
        print('starting training the model with custom training loop')
        self.train_callback.on_train_begin(self.logs)
        epoch_loss_avg = Mean()
        epoch_val_loss_avg = Mean()  # validation loss

        for epoch in range(self.parameters.get('model_epochs')):
        #for epoch, tf_dataset in self.datasets.tf_train_dataset.enumerate():
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            epoch_loss_avg.reset_states()
            epoch_val_loss_avg.reset_states()

            pbar = tqdm(enumerate(self.datasets.tf_train_dataset), desc='Loss')
            #pbar = tqdm(enumerate(tf_dataset), desc='Loss')
            if self.parameters.get('model_auxiliary'):
                for step, (in_img, in_j, in_cmd, out_of, out_aof1, out_aof2, out_aof3) in pbar:
                #for in_img, in_j, in_cmd, out_of, out_aof1, out_aof2, out_aof3 in pbar:

                    weights_predictions = self.model_fusion_weights((in_img, in_j, in_cmd))
                    # Open a GradientTape to record the operations run
                    # during the forward pass, which enables auto-differentiation.
                    with tf.GradientTape() as tape:
                        # forward pass
                        predictions = self.model((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
                        # Compute the loss value for this minibatch.
                        loss_value = self.loss_custom_loop((out_of,out_aof1,out_aof2,out_aof3), \
                                                           predictions, \
                                                           weights=weights_predictions)

                    # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))
                    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                    #pbar.set_description("Epoch loss = %f" % loss_value)
                    pbar.set_description("Epoch loss = %f" % epoch_loss_avg.result())
                    self.train_callback.on_batch_end(batch=step, logs=self.logs)

                for step, (in_img, in_j, in_cmd, out_of, out_aof1, out_aof2, out_aof3) in tqdm(enumerate(self.datasets.tf_test_dataset)):
                    weights_predictions = self.model_fusion_weights((in_img, in_j, in_cmd))
                    predictions = self.model((in_img, in_j, in_cmd), training=True)  # predictions for this minibatch
                    # Compute the loss value for this minibatch.
                    val_loss_value = self.loss_custom_loop( (out_of, out_aof1, out_aof2, out_aof3), \
                                                            predictions, \
                                                            weights=weights_predictions)
                    epoch_val_loss_avg.update_state(val_loss_value)  # Add current batch loss

            else: # model without auxiliary branches
                #for in_img, in_j, in_cmd, out_of in pbar:
                for step, (in_img, in_j, in_cmd, out_of) in pbar:
                    # Open a GradientTape to record the operations run
                    # during the forward pass, which enables auto-differentiation.
                    with tf.GradientTape() as tape:
                        # forward pass
                        predictions = self.model((in_img, in_j, in_cmd),
                                                 training=True)  # predictions for this minibatch
                        # Compute the loss value for this minibatch.
                        loss_value = self.loss_custom_loop((out_of), \
                                                           predictions)
                    #pbar.set_description("Batch loss = %f" % loss_value)
                    # Use the gradient tape to automatically retrieve
                    # the gradients of the trainable variables with respect to the loss.
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))
                    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                    pbar.set_description("Epoch loss = %f" % epoch_loss_avg.result())
                    self.train_callback.on_batch_end(batch=step, logs=self.logs)

                for step, (in_img, in_j, in_cmd, out_of) in tqdm(enumerate(self.datasets.tf_test_dataset)):
                    predictions = self.model((in_img, in_j, in_cmd))  # predictions for this minibatch
                    # Compute the loss value for this minibatch.
                    val_loss_value = self.loss_custom_loop((out_of), predictions)
                    epoch_val_loss_avg.update_state(val_loss_value)  # Add current batch loss


            self.train_callback.logs['loss']=epoch_loss_avg.result()
            self.train_callback.logs['val_loss']=epoch_val_loss_avg.result()
            print("Epoch {:03d}: Loss: {:.6f},  ValLoss: {:.6f}".format(epoch,\
                                                                        epoch_loss_avg.result(), \
                                                                        epoch_val_loss_avg.result()))
            self.train_callback.on_epoch_end(epoch=epoch)
        self.train_callback.on_train_end()
        print('training done')



    #@tf.function
    def weight_loss(self, loss_aux_mod, w, fact):
        #print('size ', str(w.numpy().shape))
        is_w_empty = tf.equal(tf.size(w), 0)
        if is_w_empty:
            print('loss weighting returns empty!!!!')
            return 0.0
        _shape = (self.parameters.get('image_size'), self.parameters.get('image_size'))
        # add dimension
        x = tf.expand_dims(w, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0]
        x = tf.tile(x, [1,_shape[1]])
        # add dimension
        x = tf.expand_dims(x, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0, image_shape_1]
        weight = tf.tile(x, [1, _shape[0], 1])
        alpha_weight = tf.math.scalar_mul(fact, tf.identity(weight))
        return loss_aux_mod * alpha_weight

    def fusion_weights_regulariser(self, loss_modality, w, fact):
        is_w_empty = tf.equal(tf.size(w), 0)
        if is_w_empty:
            print('weight regulariser returns empty!!!!')
            return 0.0
        _shape = (self.parameters.get('image_size'), self.parameters.get('image_size'))
        # add dimension
        x = tf.expand_dims(w, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0]
        x = tf.tile(x, [1, _shape[1]])
        # add dimension
        x = tf.expand_dims(x, axis=1)
        # repeat elements -> shape: [batch_size, image_shape_0, image_shape_1]
        weight = tf.tile(x, [1, _shape[0], 1])
        fact_matrix = tf.math.scalar_mul(fact, tf.ones_like(weight))
        sig_soft_loss_aux = tf.nn.softmax(tf.math.sigmoid(tf.math.exp(-tf.math.pow(loss_modality, 2))))
        return fact_matrix * tf.math.pow((weight - sig_soft_loss_aux), 2)

    #@tf.function
    def loss_custom_loop(self, y_true, y_pred, weights = []):

        true_main_out = y_true[0]
        if self.parameters.get('model_auxiliary'):
            true_aux_visual = y_true[1]
            true_aux_proprio = y_true[2]
            true_aux_motor = y_true[3]

        pred_main_out = y_pred[0]
        if self.parameters.get('model_auxiliary'):
            pred_aux_visual = y_pred[1]
            pred_aux_proprio = y_pred[2]
            pred_aux_motor = y_pred[3]

        #print ('shape pred ', str(pred_main_out.numpy().shape))
        #print ('shape true ', str(true_main_out.numpy().shape))

        #loss_main_out = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(pred_main_out), tf.squeeze(true_main_out)))
        loss_main_out = tf.keras.losses.mean_squared_error(pred_main_out, true_main_out)
        if self.parameters.get('model_auxiliary'):
            alpha = 1.0  # 0.6 is good
            beta = 0.0  # 0.0
            loss_aux_visual = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_visual), tf.squeeze(pred_aux_visual)))
            loss_aux_proprio = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_proprio), tf.squeeze(pred_aux_proprio)))
            loss_aux_motor = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(true_aux_motor), tf.squeeze(pred_aux_motor)))

            #print('loss main shape', str(loss_main_out.numpy().shape))
            #print('sss ', str(weights.numpy().shape))
            aux_loss_weighting_total = tf.reduce_mean(self.weight_loss(loss_aux_visual,  weights[:,0], alpha)) + \
                                       tf.reduce_mean(self.weight_loss(loss_aux_proprio, weights[:,1], alpha)) + \
                                       tf.reduce_mean(self.weight_loss(loss_aux_motor,   weights[:,2], alpha))

            fus_weight_regul_total = tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_visual, weights[:,0], beta)) + \
                                     tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_proprio,weights[:,1], beta)) + \
                                     tf.reduce_mean(self.fusion_weights_regulariser(loss_aux_motor,  weights[:,2], beta))

            # print('fus_weight shape true ', tf.shape(fus_weight_regulariser_total))

        #return loss_main_out# + aux_loss_weighting_total
        if self.parameters.get('model_auxiliary'):
            return loss_main_out + aux_loss_weighting_total + fus_weight_regul_total
        else:
            return loss_main_out

    def keras_training_loop(self):
        print('starting training the model with keras fit function')
        myCallback = MyCallback(self.parameters, self.datasets,self.model)
        if self.parameters.get('model_auxiliary'):
            fusion_weights_train = self.model_fusion_weights.predict( \
                [self.datasets.train_dataset_images_t, \
                 self.datasets.train_dataset_joints, \
                 self.datasets.train_dataset_cmd])

            fusion_weights_test = self.model_fusion_weights.predict( \
                [self.datasets.test_dataset_images_t, \
                 self.datasets.test_dataset_joints, \
                 self.datasets.test_dataset_cmd])

            self.history = self.model.fit([self.datasets.train_dataset_images_t, \
                                           self.datasets.train_dataset_joints, \
                                           self.datasets.train_dataset_cmd], \
                                          [self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow, \
                                           self.datasets.train_dataset_optical_flow], \
                                          epochs=self.parameters.get('model_epochs'), \
                                          batch_size=self.parameters.get('model_batch_size'), \
                                          validation_data=([self.datasets.test_dataset_images_t, \
                                                            self.datasets.test_dataset_joints, \
                                                            self.datasets.test_dataset_cmd], \
                                                           [self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow, \
                                                            self.datasets.test_dataset_optical_flow]), \
                                          shuffle=True, \
                                          callbacks=[myCallback], \
                                          verbose=1)
            #print('keras history keys ', self.history.history.keys())
        else:

            self.history=self.model.fit([self.datasets.train_dataset_images_t, \
                                         self.datasets.train_dataset_joints,\
                                         self.datasets.train_dataset_cmd],\
                           self.datasets.train_dataset_optical_flow,\
                           epochs=self.parameters.get('model_epochs'),\
                           batch_size=self.parameters.get('model_batch_size'), \
                           validation_data=([self.datasets.test_dataset_images_t, \
                                             self.datasets.test_dataset_joints, \
                                             self.datasets.test_dataset_cmd], \
                                             self.datasets.test_dataset_optical_flow),\
                           shuffle=True,\
                           callbacks = [myCallback],\
                           verbose=1)
            #print('history keys', self.history.history.keys())
        print('training done')
    '''
