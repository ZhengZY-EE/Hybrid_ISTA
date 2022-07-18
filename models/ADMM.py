'''
Description: 
Version: 1.0
Autor: Ziyang Zheng
Date: 2022-02-07 22:54:15
LastEditors: Ziyang Zheng
LastEditTime: 2022-02-11 14:47:59
'''
'''
Description: 
Version: 1.0
Autor: Ziyang Zheng
Date: 2022-02-07 22:53:58
LastEditors: Ziyang Zheng
LastEditTime: 2022-02-07 23:37:14
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of classical ISTA .
"""

import numpy as np
import tensorflow as tf
from models.LISTA_base import LISTA_base
import utils.train

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky

from utils.tf import shrink_free

class ADMM (LISTA_base):

    """
    Implementation of ADMM.
    """
    def __init__ (self, A, T, lam):
        """
        :prob:  : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this model.
        :lam    : Initial value of thresholds of shrinkage functions.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam         # this lam is euqal to config.llam, not config.lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (1.000 / self._scale).astype(np.float32)    # denote the 't' in the paper

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):

        # W = (np.transpose (self._A) / self._scale).astype (np.float32)

        self._kA_ = tf.constant (value=self._A, dtype=tf.float32)
        self._AT_ = tf.constant (value=np.transpose (self._A), dtype=tf.float32)
        self._klam = tf.constant (value=self._lam, dtype=tf.float32)
        self._t = tf.constant (value=self._theta, dtype=tf.float32)
        self._I = tf.eye(self._N)

        
    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
            
        uh_ = xh_
        zh_ = xh_ 
        xhs_.append (xh_)

        for _ in range (self._T):
            x_left = tf.matrix_inverse(tf.matmul(self._AT_, self._kA_) + self._scale * self._I)
            xh_ = tf.matmul(x_left, tf.matmul (self._AT_, y_) + self._scale * (zh_ - uh_))
            # res1_ = y_ - tf.matmul (self._kA_, xh_)
            zh_ = shrink_free (xh_ + uh_, self._klam * self._t)
            uh_ = uh_ + self._scale * (xh_ - zh_)

            xhs_.append (zh_)

        return xhs_
