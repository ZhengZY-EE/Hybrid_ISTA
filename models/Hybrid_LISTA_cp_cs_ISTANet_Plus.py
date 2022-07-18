#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Hybrid Learned ISTA with weight coupling.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base

class Hybrid_LISTA_cp_cs_ISTANet_Plus (LISTA_base):

    def __init__ (self, Phi, D, T, lam, untied, coord, scope, mode='D', conv_num=6, kernel_size=3, feature_map=32, alpha_initial=0.0):
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
        """
        self._Phi  = Phi.astype (np.float32)
        self._D    = D.astype (np.float32)
        self._A    = np.matmul (self._Phi, self._D)
        self._T    = T
        self._lam = lam
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]
        self._N    = self._D.shape [1]

        self._scale = 1.001 * np.linalg.norm (self._A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
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
        self.alphas()
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


    def alphas(self):
        alphas_ = []

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            for i in range(self._T):
                alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, initializer=self.alpha_initial ))
            self.alphas_raw = alphas_


    def setup_layers(self):
        """
        Implementation of Hybird LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in Hybird LISTA.
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

        lambda_step_list = []
        soft_thr_list = []
        # lambda_step = tf.Variable(0.1, dtype=tf.float32)
        # soft_thr = tf.Variable(0.1, dtype=tf.float32)

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

            self._PhiTPhi = tf.matmul(self._kPhi_, self._kPhi_, transpose_a=True)

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
                lambda_step_list.append(tf.get_variable(name='lambda_step_%d' % (t + 1),
                                                        dtype=tf.float32, initializer=0.1))
                soft_thr_list.append(tf.get_variable(name='soft_thr_%d' % (t + 1),
                                                     dtype=tf.float32, initializer=0.1))

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
        self.vars_in_layer = list (zip (D_inv_list[:-1], Ws1_ [:-1], Ws2_ [:-1], thetas1_ [:-1], thetas2_ [:-1], self.paras_ [:-1], self.alphas_raw [:-1], lambda_step_list[:-1], soft_thr_list[:-1]))
        self.vars_in_layer.append ((D_inv_list[-1], Ws1_ [-1], Ws2_ [-1], thetas1_ [-1], thetas2_ [-1], self.paras_ [-1], self.alphas_raw [-1],lambda_step_list[-1], soft_thr_list[-1], self._vD_, ))

        self.thetas1_ = thetas1_
        self.thetas2_ = thetas2_


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        fhs_  = [] # collection of the regressed signals

        resi_error_ = []

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
                    D_inv_, W1_, W2_, theta1_, theta2_, dnn_para_, alpha_raw, lambda_step, soft_thr = self.vars_in_layer [t]
                    D_ = self._kD_
                else:
                    D_inv_, W1_, W2_, theta1_, theta2_, dnn_para_, alpha_raw, lambda_step, soft_thr, D_ = self.vars_in_layer [t]
                 
                # v_n: [500, 64]
                res1_ = y_ - tf.matmul (self._kA_, xh_)
                vh_ = shrink_free (xh_ + tf.matmul (W1_, res1_), tf.abs(theta1_))

                # u_n: ISTANet_Plus
                un_ista1 = tf.matmul (D_, vh_)    # [16 * 16, bs]
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
                
                # u_n end
                uh_ = tf.matmul(D_inv_, tf.transpose(tf.reshape(un_ista11, [-1, 16*16])))
  
                # w_n
                res2_ = y_ - tf.matmul (self._kA_, uh_)
                wh_ = shrink_free (uh_ + tf.matmul (W2_, res2_), tf.abs(theta2_))

                # 1-alpha_n (0, 1 - lower bound of alpha)
                one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(theta1_) / (tf.abs(theta1_)+tf.abs(theta2_))

                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_

                xhs_.append (xh_)

                fhs_.append (tf.matmul (D_, xh_))

        return xhs_, fhs_, resi_error_
