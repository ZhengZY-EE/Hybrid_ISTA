'''
Description: ELISTA.
Version: 1.0
Autor: Ziyang Zheng
Date: 2021-11-12 18:54:06
LastEditors: Ziyang Zheng
LastEditTime: 2022-01-26 16:23:45
'''
import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import MT, shrink_free
from models.LISTA_base import LISTA_base

class ELISTA_cs (LISTA_base):

    """
    Implementation of ELISTA_cs.
    """
    def __init__ (self, Phi, D, T, lam, untied, coord, scope, mt_flag):
        """
        :prob:  : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this LISTA model.
        :lam    : Initial value of thresholds of MT.
        :lam_bar: Initial value of thresholds_bar of MT.
        :untied : Whether weights are shared within layers.
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
        self._mt_flag = mt_flag

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
        Ws1_     = []
        Ws2_     = []
        alphas1_  = []
        alphas2_  = []
        thetas1_ = []
        thetas2_ = []

        W = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._kD_   = tf.constant (value=self._D, dtype=tf.float32)
            self._kA_   = tf.constant (value=self._A, dtype=tf.float32)
            self._vD_   = tf.get_variable (name='D', dtype=tf.float32,
                                           initializer=self._D)

            if not self._untied: # tied model
 
                Ws1_.append (tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=W))
                Ws1_ = Ws1_ * self._T
                Ws2_ = Ws1_
                
                for t in range (self._T):
                    alphas1_.append (tf.get_variable (name="alpha1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                    alphas2_.append (tf.get_variable (name="alpha2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
            else:
                
                alphas1_.append (tf.get_variable (name="alpha",
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                alphas1_ = alphas1_ * self._T
                alphas2_ = alphas1_
                
                for t in range (self._T):
                    Ws1_.append (tf.get_variable (name="W1_%d"%(t+1),
                                                        dtype=tf.float32,
                                                        initializer=W))
                    Ws2_.append (tf.get_variable (name="W2_%d"%(t+1),
                                                        dtype=tf.float32,
                                                        initializer=W))

            if not self._mt_flag:
                for t in range (self._T):
                    thetas1_.append (tf.get_variable (name="theta1_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))
                    thetas2_.append (tf.get_variable (name="theta2_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))    
                         
                self.vars_in_layer = list (zip (Ws1_[:-1], Ws2_[:-1], alphas1_[:-1], alphas2_[:-1], thetas1_[:-1], thetas2_[:-1])) 
                self.vars_in_layer.append((Ws1_[-1], Ws2_[-1], alphas1_[-1], alphas2_[-1], thetas1_[-1], thetas2_[-1], self._vD_, ))
                  
            else:
                thetas1_bar_ = []
                thetas2_bar_ = []
                for t in range (self._T):
                    thetas1_.append (tf.get_variable (name="theta1_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta - 0.1).astype(np.float32)))
                    thetas2_.append (tf.get_variable (name="theta2_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta - 0.1).astype(np.float32)))  
                    thetas1_bar_.append (tf.get_variable (name="theta1_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))
                    thetas2_bar_.append (tf.get_variable (name="theta2_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))  
                
                self.vars_in_layer = list (zip (Ws1_[:-1], Ws2_[:-1], alphas1_[:-1], alphas2_[:-1], thetas1_[:-1], thetas2_[:-1], thetas1_bar_[:-1], thetas2_bar_[:-1])) 
                self.vars_in_layer.append((Ws1_[-1], Ws2_[-1], alphas1_[-1], alphas2_[-1], thetas1_[-1], thetas2_[-1], thetas1_bar_[-1], thetas2_bar_[-1], self._vD_, ))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        # self.vars_in_layer = list (zip (Ws_, thetas_))


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
            if not self._mt_flag:
                for t in range (self._T):
                    if t < self._T - 1:
                        W1_, W2_, alpha1_, alpha2_, theta1_, theta2_ = self.vars_in_layer [t]
                        D_ = self._kD_
                    else:
                        W1_, W2_, alpha1_, alpha2_, theta1_, theta2_, D_ = self.vars_in_layer [t]

                    res1_ = y_ - tf.matmul (self._kA_, xh_)
                    xh_mid_ = shrink_free (xh_ + alpha1_ * tf.matmul (W1_, res1_), theta1_)
                    res2_ = y_ - tf.matmul (self._kA_, xh_mid_)
                    xh_ = shrink_free (xh_ + alpha2_ * tf.matmul (W2_, res2_), theta2_)
                    xhs_.append (xh_)
                    fhs_.append (tf.matmul (D_, xh_))
            else:
                for t in range (self._T):
                    if t < self._T - 1:
                        W1_, W2_, alpha1_, alpha2_, theta1_, theta2_, theta1_bar_, theta2_bar_ = self.vars_in_layer [t]
                        D_ = self._kD_
                    else:
                        W1_, W2_, alpha1_, alpha2_, theta1_, theta2_, theta1_bar_, theta2_bar_, D_ = self.vars_in_layer [t]

                    res1_ = y_ - tf.matmul (self._kA_, xh_)
                    xh_mid_ = MT (xh_ + alpha1_ * tf.matmul (W1_, res1_), theta1_, theta1_bar_)
                    res2_ = y_ - tf.matmul (self._kA_, xh_mid_)
                    xh_ = MT (xh_ + alpha2_ * tf.matmul (W2_, res2_), theta2_, theta2_bar_)
                    xhs_.append (xh_)
                    fhs_.append (tf.matmul (D_, xh_))
        return xhs_, fhs_