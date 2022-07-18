#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Hybrid classical ISTA .
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base

class Hybrid_ISTA_freeT (LISTA_base):

    """
    Implementation of Hybrid classical ISTA .
    """
    def __init__ (self, A, T, lam, untied, coord, scope, mode='S', conv_num=3, kernel_size=9, feature_map=16, alpha_initial=0.0):
        """
        :prob:  : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this Hybrid LISTA model.
        :lam    : Initial value of thresholds of shrinkage functions.
        :untied : Whether weights are shared within layers.
                  If tied, W1, W2 in all iteration are shared and DNNs between different iterations are the same. Parameters: [DNNs, W]
                  If untied, please refer to option 'mode'.
        :mode   : Decide whether two weights are shared. Theta1, Theta2 and Alpha are always not shared.
                  'D': Different. No parameters are shared. Parameters: [DNNs, W1, W2] * T
                  'S': Same. W1 and W2 in one iteration are the same. Parameters: [DNNs, W] * T
        In this model, only untied and tied with 'S' are executable.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam         # this lam is euqal to config.llam, not config.lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self.upper = (1.000 / np.linalg.norm (A, ord=2)**2).astype(np.float32) 

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (1.000 / self._scale).astype(np.float32)    # denote the 't' in the paper
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope
        self._mode = mode

        self.conv_num = conv_num
        self.kernel_size = kernel_size
        self.feature_map = feature_map
        self.alpha_initial = alpha_initial

        """ Set up layers."""
        self.dnns_paras()
        # self.alphas()
        self.setup_layers()


    def dnns_paras(self):
        '''
        Parameters of DNNs. Depends on the DNN architecture. You can define it whatever you like.
        '''
        paras_ = []
        paras_total_ = []
        
        assert self.conv_num > 1 
        assert self.kernel_size > 0
        assert self.feature_map > 0

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            if not self._untied: # tied model
                for i in range(self.conv_num):
                    if i == 0:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, 1, self.feature_map], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                    elif i == self.conv_num-1:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, self.feature_map, 1], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                    else:
                        paras_.append (tf.get_variable (name='conv_'+str(i+1), 
                                                        shape=[self.kernel_size, self.feature_map, self.feature_map], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                paras_total_.append(paras_)
                self.paras_ = paras_total_ * self._T

            if self._untied: # untied model
                for j in range(self._T):
                    for i in range(self.conv_num):
                        if i == 0:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, 1, self.feature_map], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                        elif i == self.conv_num-1:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, self.feature_map, 1], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                        else:
                            paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_'+str(i+1), 
                                                            shape=[self.kernel_size, self.feature_map, self.feature_map], dtype=tf.float32,
                                                            initializer=tf.orthogonal_initializer()))
                    paras_total_.append(paras_)
                    paras_ = []
                self.paras_ = paras_total_


    def alphas(self):
        alphas_ = []

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            for i in range(self._T):
                alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, 
                                                initializer=self.alpha_initial))

                # alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, 
                #                                 initializer=tf.random_uniform(-5, 5)))
            self.alphas_raw = alphas_


    def setup_layers(self):
        ts_scalar_ = []

        # W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)
            self._AT_ = tf.constant (value=np.transpose (self._A), dtype=tf.float32)
            self._upper = tf.constant (value=self.upper, dtype=tf.float32)

            if self._mode != 'S':
                print('No such name of mode. In this model, only S is executable.')
                raise ValueError

            for t in range (self._T):
                ts_scalar_.append (tf.get_variable (name="t_scalar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))

                # delta_total_.append(tf.get_variable (name="delta_%d"%(t+1),
                #                                     dtype=tf.float32,
                #                                     initializer=tf.random_uniform(-5, 5)))     
                        
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (ts_scalar_, self.paras_))
        self.thetas_ = ts_scalar_


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        self.eta_list = []

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                
                t_, dnn_para_ = self.vars_in_layer [t]
                 
                # v_n: [500, 64]
                res1_ = y_ - tf.matmul (self._kA_, xh_)
                vh_ = shrink_free (xh_ + tf.abs(t_) * tf.matmul (self._AT_, res1_), self._lam * tf.abs(t_))

                # u_n
                vh_0 = tf.expand_dims(tf.transpose(vh_), -1)
                
                for i in range(self.conv_num):
                    if i == self.conv_num - 1:
                        vh_0 = tf.nn.conv1d(vh_0, dnn_para_[i], stride=1, padding='SAME')
                    else: 
                        vh_0 = tf.nn.relu(tf.nn.conv1d(vh_0, dnn_para_[i], stride=1, padding='SAME'))

                uh_ = tf.reshape(vh_0, tf.shape(vh_)) + vh_

                self.eta_list.append(tf.norm(uh_-xh_)/tf.norm(vh_-xh_))
  
                # w_n
                res2_ = y_ - tf.matmul (self._kA_, uh_)
                wh_ = shrink_free (uh_ + tf.abs(t_) * tf.matmul (self._AT_, res2_), self._lam * tf.abs(t_))

                # 1-alpha_n (0, 1 - lower bound of alpha)
                # tu_x = tf.square(tf.norm(uh_-xh_))
                # tv_x = (1-2 * t_ * del_  / self.upper) * tf.square(tf.norm(vh_-xh_))
                # one_alpha_ = tf.sigmoid(alpha_raw) * tv_x / (tu_x + tv_x)
                one_alpha_ = 0.5
                
                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                xhs_.append (xh_)
            
        return xhs_
