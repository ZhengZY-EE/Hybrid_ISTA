#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base

class Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus (LISTA_base):

    def __init__ (self, Phi, D, T, lam, untied, coord, scope, mode='S', conv_num=6, kernel_size=3, feature_map=32, alpha_initial=0.0):
        
        self._Phi  = Phi.astype (np.float32)
        self._D    = D.astype (np.float32)
        self._A    = np.matmul (self._Phi, self._D)
        self._T   = T
        ####################
        self._lam = lam      
        ####################   
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]
        self._N    = self._D.shape [1]

        self.upper = (1.000 / np.linalg.norm (self._A, ord=2)**2).astype(np.float32) 

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
        delta_total_ = []
        ts_scalar_ = []
        lams_ = []
        D_inv_list = []

        lambda_step_list = []
        soft_thr_list = []

        # W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._kD_   = tf.constant (value=self._D, dtype=tf.float32)
            self._kA_   = tf.constant (value=self._A, dtype=tf.float32)
            self._vD_   = tf.get_variable (name='D', dtype=tf.float32,
                                           initializer=self._D)
            self._upper = tf.constant (value=self.upper, dtype=tf.float32)
            self._AT_ = tf.constant (value=np.transpose (self._A), dtype=tf.float32)
            self._D_inv = tf.get_variable (name='D_inv',shape=[self._N, self._F],
                                           dtype=tf.float32, initializer=tf.orthogonal_initializer())
            D_inv_list.append(self._D_inv)
            D_inv_list = D_inv_list * self._T

            self._PhiTPhi = tf.matmul(self._kPhi_, self._kPhi_, transpose_a=True)
            
            if self._mode != 'S':
                print('No such name of mode. In this model, only S is executable.')
                raise ValueError

            for t in range (self._T):
                ts_scalar_.append (tf.get_variable (name="t_scalar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))
                delta_total_.append(tf.get_variable (name="delta_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=0.0)) 
                lams_.append(tf.get_variable(name="lam_%d"%(t+1),
                                            dtype=tf.float32,
                                            initializer=self._lam))

                lambda_step_list.append (tf.get_variable (name='lambda_step_%d'%(t+1), dtype=tf.float32, initializer=0.1))
                soft_thr_list.append (tf.get_variable (name='soft_thr_%d'%(t+1), dtype=tf.float32, initializer=0.1))
                        
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (delta_total_[:-1], D_inv_list[:-1], ts_scalar_ [:-1], self.paras_ [:-1], self.alphas_raw [:-1], lams_[:-1], lambda_step_list[:-1], soft_thr_list[:-1]))
        self.vars_in_layer.append ((delta_total_[-1], D_inv_list[-1], ts_scalar_ [-1], self.paras_ [-1], self.alphas_raw [-1], lams_[-1], lambda_step_list[-1], soft_thr_list[-1], self._vD_, ))
        self.ts_ = ts_scalar_

    def lam_range(self, lam_input, x_n, x_n_1, lam_old, C=1):
        Q = C * tf.norm(x_n-x_n_1)
        P = tf.cond(Q < lam_old, lambda: Q, lambda: lam_old)
        lam_final = tf.sigmoid(lam_input) * P * 1.89    # 1.89 for lam_init = 0.1
        return lam_final

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
                    del_, D_inv_, t_, dnn_para_, alpha_raw, lam_, lambda_step, soft_thr = self.vars_in_layer [t]
                    D_ = self._kD_
                else:
                    del_, D_inv_, t_, dnn_para_, alpha_raw, lam_, lambda_step, soft_thr, D_ = self.vars_in_layer [t]

                del_ = tf.sigmoid(del_) * 0.25 + 0.25
        
                t_ = tf.clip_by_value(t_, self._upper/(4*del_), self._upper)

                if t == 0:
                    lam_f = lam_
                else:
                    lam_f = self.lam_range(lam_, xh_, xhs_[t-1], lam_old)

                lam_old = lam_f
                 
                # v_n
                res1_ = y_ - tf.matmul (self._kA_, xh_)
                vh_ = shrink_free (xh_ + tf.abs(t_) * tf.matmul (self._AT_, res1_), lam_f * tf.abs(t_))

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
                wh_ = shrink_free (uh_ + tf.abs(t_) * tf.matmul (self._AT_, res2_), lam_f * tf.abs(t_))

                # 1-alpha_n (0, 1 - lower bound of alpha)
                tu_x = tf.square(tf.norm(uh_-xh_))
                tv_x = (1-2 * t_ * del_  / self._upper) * tf.square(tf.norm(vh_-xh_))
                one_alpha_ = tf.sigmoid(alpha_raw) * tv_x / (tu_x + tv_x)

                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                xhs_.append (xh_)

                fhs_.append (tf.matmul (D_, xh_))

        return xhs_, fhs_, resi_error_
