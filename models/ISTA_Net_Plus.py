#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Hybrid Learned ISTA with weight coupling.
"""

import numpy as np
import tensorflow as tf
import utils.train

from models.LISTA_base import LISTA_base

class ISTA_Net_Plus (LISTA_base):

    def __init__ (self, Phi, T, untied, scope, conv_num=6, kernel_size=3, feature_map=32):
        """
        :prob:  : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this Hybrid LISTA model.
        :untied : Whether weights are shared within layers.
                  If tied, W1, W2 in all iteration are shared and DNNs between different iterations are the same. Parameters: [DNNs, W]
                  If untied, please refer to option 'mode'.
        """
        self._Phi  = Phi.astype (np.float32)
        self._T    = T
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]

        self._untied = untied
        self._scope  = scope

        self.conv_num = conv_num
        self.kernel_size = kernel_size
        self.feature_map = feature_map

        """ Set up layers."""
        self.dnns_paras()
        self.setup_layers()


    def dnns_paras(self):
        '''
        Parameters of DNNs. Depends on the DNN architecture. You can define it whatever you like.
        '''
        paras_ = []
        paras_total_ = []
        
        assert self.conv_num == 6 
        assert self.kernel_size == 3 
        assert self.feature_map == 32

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            if not self._untied: # tied model
                for i in range(self.conv_num):
                    if i == 0:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, self.kernel_size, 1, self.feature_map], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                    elif i == self.conv_num-1:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, self.kernel_size, self.feature_map, 1], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                    else:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, self.kernel_size, self.feature_map, self.feature_map], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                paras_total_.append(paras_)
                self.paras_ = paras_total_ * self._T

            if self._untied: # untied model
                for j in range(self._T):
                    for i in range(self.conv_num):
                        if i == 0:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, self.kernel_size, 1, self.feature_map], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                        elif i == self.conv_num-1:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, self.kernel_size, self.feature_map, 1], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                        else:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, self.kernel_size, self.feature_map, self.feature_map], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                    paras_total_.append(paras_)
                    paras_ = []
                self.paras_ = paras_total_


    def setup_layers(self):

        lambda_step_list = []
        soft_thr_list = []

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._PhiTPhi = tf.matmul(self._kPhi_, self._kPhi_, transpose_a=True)

            if not self._untied: # tied model
                lambda_step_list.append (tf.get_variable (name='lambda_step', dtype=tf.float32, initializer=0.1))
                lambda_step_list = lambda_step_list * self._T
                soft_thr_list.append (tf.get_variable (name='soft_thr', dtype=tf.float32, initializer=0.1))
                soft_thr_list = soft_thr_list * self._T

            if self._untied: # untied model
                for t in range (self._T):
                    lambda_step_list.append (tf.get_variable (name='lambda_step_%d'%(t+1), dtype=tf.float32, initializer=0.1))
                    soft_thr_list.append (tf.get_variable (name='soft_thr_%d'%(t+1), dtype=tf.float32, initializer=0.1))

        self.vars_in_layer = list (zip (self.paras_, lambda_step_list, soft_thr_list))


    def inference (self, y_, x0_=None):
        xhs_  = [] 
        resi_error_ = []

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._F, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                dnn_para_, lambda_step, soft_thr = self.vars_in_layer [t]
        
                # u_n: ISTANet_Plus
                un_ista1 = xh_
                un_ista2 = tf.add(un_ista1 - tf.scalar_mul(lambda_step, tf.matmul(self._PhiTPhi, un_ista1)),
                        tf.scalar_mul(lambda_step, tf.matmul(self._kPhi_, y_, transpose_a=True)))  # X_k - lambda*A^TAX
                un_ista3 = tf.reshape(tf.transpose(un_ista2), [-1, 16, 16, 1])   # [-1, 16, 16, 1]
                un_ista4 = tf.nn.conv2d(un_ista3, dnn_para_[0], strides=[1, 1, 1, 1], padding='SAME')
                un_ista5 = tf.nn.relu(tf.nn.conv2d(un_ista4, dnn_para_[1], strides=[1, 1, 1, 1], padding='SAME'))
                un_ista6 = tf.nn.conv2d(un_ista5, dnn_para_[2], strides=[1, 1, 1, 1], padding='SAME')
                un_ista7 = tf.multiply(tf.sign(un_ista6), tf.nn.relu(tf.abs(un_ista6) - soft_thr))
                un_ista8 = tf.nn.relu(tf.nn.conv2d(un_ista7, dnn_para_[3], strides=[1, 1, 1, 1], padding='SAME'))
                un_ista9 = tf.nn.conv2d(un_ista8, dnn_para_[4], strides=[1, 1, 1, 1], padding='SAME')
                un_ista10 = tf.nn.conv2d(un_ista9, dnn_para_[5], strides=[1, 1, 1, 1], padding='SAME')
                un_ista11 = un_ista10 + un_ista3   # [-1, 16, 16, 1]

                # identical operator
                un_ista8_sys = tf.nn.relu(tf.nn.conv2d(un_ista6, dnn_para_[3], strides=[1, 1, 1, 1], padding='SAME'))
                un_ista9_sys = tf.nn.conv2d(un_ista8_sys, dnn_para_[4], strides=[1, 1, 1, 1], padding='SAME')
                resi_error_.append(un_ista9_sys - un_ista4)

                # Block End
                xh_ = tf.transpose(tf.reshape(un_ista11, [-1, 16*16]))
                xhs_.append (xh_)

        return xhs_, resi_error_
