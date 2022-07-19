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

class Hybrid_ISTA_ada_freeT_FixNetFunc (LISTA_base):

    """
    Implementation of Hybrid classical ISTA .
    """
    def __init__ (self, A, T, lam, untied, coord, scope, mode='S', alpha_initial=0.0, Net_Func='0'):

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

        self.alpha_initial = alpha_initial
        self.Net_Func = Net_Func
        
        """ Set up layers."""
        self.setup_layers()


    def FixFunc(self, x):
        if self.Net_Func == '0':
            y = tf.zeros_like(x)
        elif self.Net_Func == 'x':
            y = x
        elif self.Net_Func == 'x^2':
            y = tf.pow(x, 2)
        # elif self.Net_Func == 'ln_x':
        #     y = tf.log(x)
        elif self.Net_Func == 'e^x':
            y = tf.exp(x)
        else:
            print('The specified function is not defined.')
            raise ValueError
        return y


    def alphas(self):
        alphas_ = []

        with tf.variable_scope (self._scope, reuse=False) as vs: 
            for i in range(self._T):
                alphas_.append(tf.get_variable (name="alpha_%d"%(i+1), dtype=tf.float32, 
                                                initializer=self.alpha_initial))
            self.alphas_raw = alphas_


    def setup_layers(self):
        ts_scalar_ = []
        lams_ = []

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
         
                lams_.append(tf.get_variable(name="lam_%d"%(t+1),
                                            dtype=tf.float32,
                                            initializer=self._lam))
                        
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (lams_, ts_scalar_))
        self.thetas_ = ts_scalar_
        self.lams_ = lams_


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
                
                lam_, t_ = self.vars_in_layer [t]
                 
                # v_n: [500, 64]
                res1_ = y_ - tf.matmul (self._kA_, xh_)
                vh_ = shrink_free (xh_ + tf.abs(t_) * tf.matmul (self._AT_, res1_), lam_ * tf.abs(t_))

                # u_n
                uh_ = self.FixFunc(vh_)  
                self.eta_list.append(tf.norm(uh_-xh_)/tf.norm(vh_-xh_))

                # w_n
                res2_ = y_ - tf.matmul (self._kA_, uh_)
                wh_ = shrink_free (uh_ + tf.abs(t_) * tf.matmul (self._AT_, res2_), lam_ * tf.abs(t_))

                # 1-alpha_n (0, 1 - lower bound of alpha)
                # tu_x = tf.square(tf.norm(uh_-xh_))
                # tv_x = (1-2 * t_ * del_  / self.upper) * tf.square(tf.norm(vh_-xh_))
                # one_alpha_ = tf.sigmoid(alpha_raw) * tv_x / (tu_x + tv_x)
                one_alpha_ = 0.5
                
                # x_n+1
                xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                xhs_.append (xh_)

        return xhs_
