#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LISTA_cpss_corrected.py
author: xhchrn
        chernxh@tamu.edu
date  : 2018-10-21

Implementation of Learned ISTA with support selection and coupled weights like
in LAMP, without setting thresholds to zeros if they are minor to zero.
"""

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import hard_shrink
from models.LISTA_base import LISTA_base

class LIHT (LISTA_base):

    """
    Implementation of deep neural network model.
    """

    def __init__ (self, A, T, lam, untied, coord, scope):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam       : Initial value of thresholds of shrinkage functions.
        :untied    : Whether weights are shared within layers.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._theta = np.sqrt (self._lam)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


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
        Bs_     = []
        Ws_     = []
        thetas_ = []

        B = (np.transpose (self._A)).astype (np.float32)
        W = np.eye (self._N, dtype=np.float32) - np.matmul (B, self._A)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            Bs_.append (tf.get_variable (name='B', dtype=tf.float32,
                                         initializer=B))
            Bs_ = Bs_ * self._T
            if not self._untied: # tied model
                Ws_.append (tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=W))
                Ws_ = Ws_ * self._T

            for t in range (self._T):
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name="W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=W))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_, Ws_, thetas_))


    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_, W_, theta_ = self.vars_in_layer [t]

                By_ = tf.matmul (B_, y_)
                xh_ = hard_shrink (tf.matmul (W_, xh_) + By_, theta_)
                xhs_.append (xh_)

        return xhs_

