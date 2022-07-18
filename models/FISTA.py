'''
Description: 
Version: 1.0
Autor: Ziyang Zheng
Date: 2022-02-07 22:54:05
LastEditors: Ziyang Zheng
LastEditTime: 2022-02-08 00:56:02
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
Implementation of classical FISTA .
"""

import numpy as np
import tensorflow as tf
from models.LISTA_base import LISTA_base
import utils.train

from utils.tf import shrink_free

class FISTA (LISTA_base):

    """
    Implementation of classical FISTA .
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
            
        
    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes
        t_current = 0.0

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
            
        yh_ = xh_
        xhs_.append (xh_)

        for _ in range (self._T):
            
            xh_old = xh_
            
            res1_ = y_ - tf.matmul (self._kA_, yh_)
            
            xh_ = shrink_free (yh_ + self._t * tf.matmul (self._AT_, res1_), self._klam * self._t)
            
            t_new = (1.0 + tf.sqrt(1.0 + 4.0 * tf.square(t_current))) / 2.0
            
            yh_ = xh_ + (t_current - 1) / (t_new) * (xh_ - xh_old)            
            
            t_current = t_new

            xhs_.append (xh_)

        return xhs_
