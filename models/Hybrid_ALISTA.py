#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Hybrid ALISTA.
"""

import numpy as np
import tensorflow as tf

from utils.tf import shrink_ss, is_tensor, shrink_free
from models.LISTA_base import LISTA_base


class Hybrid_ALISTA(LISTA_base):

    """
    Implementation of Hybrid ALISTA.
    """

    def __init__(self, A, T, lam, W, percent, max_percent, untied, coord, scope, 
                 mode='D', conv_num=3, kernel_size=9, feature_map=16, alpha_initial=0.0):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam  : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers. In this model, only apply to the DNN parameters.
        """
        self._A    = A.astype(np.float32)
        self._W    = W
        self._T    = T
        self._p    = percent
        self._maxp = max_percent
        self._lam  = lam
        self._M    = self._A.shape[0]
        self._N    = self._A.shape[1]

        self._scale = 1.001 * np.linalg.norm(A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t+1) * self._p for t in range(self._T)]
        self._ps = np.clip(self._ps, 0.0, self._maxp)

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
        self.alphas()
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
                alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, initializer=self.alpha_initial ))
            self.alphas_raw = alphas_


    def setup_layers(self):
        """ Set up layers of ALISTA.
        """
        gamas1_ = [] # step sizes
        thetas1_ = [] # thresholds
        thetas2_ = []

        with tf.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant(value=self._A, dtype=tf.float32)
            if not is_tensor(self._W):
                self._W_ = tf.constant(value=self._W, dtype=tf.float32)
            else:
                self._W_ = self._W
            self._Wt_ = tf.transpose(self._W_, perm=[1,0])


            for t in range(self._T):
                thetas1_.append(tf.get_variable(name="theta1_%d"%(t+1),
                                                dtype=tf.float32,
                                                initializer=self._theta))
                thetas2_.append(tf.get_variable(name="theta2_%d"%(t+1),
                                                dtype=tf.float32,
                                                initializer=self._theta))
                gamas1_.append(tf.get_variable (name="gama1_%d"%(t+1),
                                                dtype=tf.float32,
                                                initializer=1.0))
                

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list(zip(gamas1_, thetas1_, thetas2_, self.paras_, self.alphas_raw))
        self.thetas1_ = thetas1_
        self.thetas2_ = thetas2_


    def inference(self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        self.uhs_ = []

        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append(xh_)

        with tf.variable_scope(self._scope, reuse=True) as vs:
            for t in range(self._T):
                gama1_, theta1_, theta2_, dnn_para_, alpha_raw = self.vars_in_layer[t]
                # percent = self._ps[t]

                # v_n
                res1_ = y_ - tf.matmul(self._kA_, xh_)
                zh1_ = xh_ + gama1_ * tf.matmul(self._Wt_, res1_)
                # vh_ = shrink_ss(zh1_, tf.abs(theta1_), percent)
                vh_ = shrink_free(zh1_, tf.abs(theta1_))

                # u_n
                vh_0 = tf.expand_dims(tf.transpose(vh_), -1)
                
                for i in range(self.conv_num):
                    if i == self.conv_num - 1:
                        vh_0 = tf.nn.conv1d(vh_0, dnn_para_[i], stride=1, padding='SAME')
                    else: 
                        vh_0 = tf.nn.relu(tf.nn.conv1d(vh_0, dnn_para_[i], stride=1, padding='SAME'))

                uh_ = tf.reshape(vh_0, tf.shape(vh_)) + vh_

                # w_n
                res2_ = y_ - tf.matmul(self._kA_, uh_)
                zh2_ = uh_ + tf.matmul(self._Wt_, res2_)
                # wh_ = shrink_ss(zh2_, tf.abs(theta2_), percent)
                wh_ = shrink_free(zh2_, tf.abs(theta2_))

                # 1-alpha_n (0, 1 - lower bound of alpha)
                one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(theta1_) / (tf.abs(theta1_)+tf.abs(theta2_))

                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                xhs_.append(xh_)
                self.uhs_.append(uh_)

        return xhs_


