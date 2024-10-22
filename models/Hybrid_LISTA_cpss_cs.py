#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_ss
from models.LISTA_base import LISTA_base


class Hybrid_LISTA_cpss_cs (LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__(self, Phi, D, T, lam, percent, max_percent,
                 untied, coord, scope, mode='D', conv_num=3, kernel_size=3, feature_map=16, alpha_initial=0.0):
        
        self._Phi  = Phi.astype (np.float32)
        self._D    = D.astype (np.float32)
        self._A    = np.matmul (self._Phi, self._D)
        self._T    = T
        self._p    = percent
        self._maxp = max_percent
        self._lam  = lam
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]
        self._N    = self._D.shape [1]

        self._scale = 1.001 * np.linalg.norm (self._A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._ps = [(t+1) * self._p for t in range (self._T)]
        self._ps = np.clip (self._ps, 0.0, self._maxp)

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


    def alphas(self):
        alphas_ = []

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            for i in range(self._T):
                alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, initializer=self.alpha_initial ))
            self.alphas_raw = alphas_


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Ws1_    = []
        Ws2_    = []
        thetas1_ = []
        thetas2_ = []
        D_inv_list = []

        W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._kD_   = tf.constant (value=self._D, dtype=tf.float32)
            self._kA_   = tf.constant (value=self._A, dtype=tf.float32)
            self._vD_   = tf.get_variable (name='D', dtype=tf.float32,
                                           initializer=self._D)
            self._D_inv = tf.get_variable (name='D_inv',shape=[self._N, self._F],
                                           dtype=tf.float32, initializer=tf.orthogonal_initializer())
            D_inv_list.append(self._D_inv)
            D_inv_list = D_inv_list * self._T

            if not self._untied: # tied model
                Ws1_.append (tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=W))
                Ws1_ = Ws1_ * self._T
                Ws2_ = Ws1_

            for t in range (self._T):
                thetas1_.append (tf.get_variable (name="theta1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                thetas2_.append (tf.get_variable (name="theta2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    if self._mode == 'D':
                        Ws1_.append (tf.get_variable (name="W1_%d"%(t+1),
                                                      dtype=tf.float32,
                                                      initializer=W))
                        Ws2_.append (tf.get_variable (name="W2_%d"%(t+1),
                                                      dtype=tf.float32,
                                                      initializer=W))
                    elif self._mode == 'S':
                        Ws1_.append (tf.get_variable (name="W_%d"%(t+1),
                                                      dtype=tf.float32,
                                                      initializer=W))
                        Ws2_.append(Ws1_[t])
                    else:
                        print('No such name of mode.')
                        raise ValueError

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        # Note here the last element of `self.vars_in_layer` is
        # (W_, theta_, vD_)
        self.vars_in_layer = list (zip (D_inv_list[:-1], Ws1_ [:-1], Ws2_ [:-1], thetas1_ [:-1], thetas2_ [:-1], self.paras_ [:-1], self.alphas_raw [:-1]))
        self.vars_in_layer.append ((D_inv_list[-1], Ws1_ [-1], Ws2_ [-1], thetas1_ [-1], thetas2_ [-1], self.paras_ [-1], self.alphas_raw [-1], self._vD_, ))

        self.thetas1_ = thetas1_
        self.thetas2_ = thetas2_

    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        fhs_  = [] # collection of the regressed signals

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)
        fhs_.append (tf.matmul (self._kD_, xh_))

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                if t < self._T - 1:
                    D_inv_, W1_, W2_, theta1_, theta2_, dnn_para_, alpha_raw = self.vars_in_layer [t]
                    D_ = self._kD_
                else:
                    D_inv_, W1_, W2_, theta1_, theta2_, dnn_para_, alpha_raw, D_ = self.vars_in_layer [t]
                percent = self._ps [t]

                # v_n
                res1_ = y_ - tf.matmul (self._kA_, xh_)
                vh_ = shrink_ss (xh_ + tf.matmul (W1_, res1_), tf.abs(theta1_), percent)

                # u_n
                vh_0 = tf.reshape(tf.transpose(tf.matmul (D_, vh_)), [-1, 16, 16, 1])
                # vh_0 = tf.expand_dims(tf.transpose(vh_), -1)
                
                for i in range(self.conv_num):
                    if i == self.conv_num - 1:
                        vh_0 = tf.nn.conv2d(vh_0, dnn_para_[i], strides=[1,1,1,1], padding='SAME')
                    else: 
                        vh_0 = tf.nn.relu(tf.nn.conv2d(vh_0, dnn_para_[i], strides=[1,1,1,1], padding='SAME'))
                
                vh_0 = tf.matmul(D_inv_, tf.transpose(tf.reshape(vh_0, [-1, 16*16])))
                uh_ = vh_0 + vh_
  
                # w_n
                res2_ = y_ - tf.matmul (self._kA_, uh_)
                wh_ = shrink_ss (uh_ + tf.matmul (W2_, res2_), tf.abs(theta2_), percent)

                # 1-alpha_n (0, 1 - lower bound of alpha)
                one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(theta1_) / (tf.abs(theta1_)+tf.abs(theta2_))

                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_

                xhs_.append (xh_)

                fhs_.append (tf.matmul (D_, xh_))

        return xhs_, fhs_

