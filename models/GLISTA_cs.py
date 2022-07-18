'''
Description: Gated LISTA
Version: 1.0
Autor: https://github.com/wukailun/GLISTA/blob/master/GLISTA_cp.py
Date: 2021-11-15 14:18:40
LastEditors: Ziyang Zheng
LastEditTime: 2022-01-26 16:13:00
'''
import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base


### This in the main coding of the GLISTA with couple parameters
class GLISTA_cs (LISTA_base):
    
    """
    Implementation of GLISTA_cs.
    Old Version:
    1) Always set theta as a vector.
    2) Always utilize shrink_ss.
    3) v_t in gain functions is set to a vector, which is not consistant with the paper.
    4) The sigmoid-based function for overshoot gate is not consistant with Eq.(18) in the paper.
    5) The overshoot gate is wrong in the line 233.
    """
    def __init__(self, Phi, D, T, lam, untied, coord, scope, alti, overshoot, gain_fun, over_fun, both_gate, T_combine, T_middle):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam       : Initial value of thresholds of MT.
        :untied    : Whether weights are shared within layers.
        :alti      : Initial mu_t in gain functions.
        :overshoot : Whether we adopt overshoot gate.
        :gain_fun  : The name of gain functions, [relu, inv, exp, sigm, inv_v, combine, none].
        :over_fun  : The name of overshoot functions, [inv, sigm, none].
        :both_gate : Whether we utilize both gates in each layer.
        :T_combine : When 'both_gate' & 'gain_fun == 'combine', it represents the flag layer for adopting different gain_fun.
        :T_middle  : When no 'both_gate', it represents the flag layer for only adopting gain_fun or over_fun.
        """
        
        self._Phi  = Phi.astype (np.float32)
        self._D    = D.astype (np.float32)
        self._A    = np.matmul (self._Phi, self._D)
        self._T    = T
        self._lam = lam
        self._M    = self._Phi.shape [0]
        self._F    = self._Phi.shape [1]
        self._N    = self._D.shape [1]
        
        
        self._overshoot = overshoot 
        self._scale = 1.001 * np.linalg.norm (self._A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        self._alti = alti
        
        # (Old Version) We set theta as a vector forever 
        # For fair comparison, we set 'theta' as a scalar when 'no_coord'.
        # self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta
        
        self._logep = -2.0
        
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta
            
        self.gain_gate = []
        self.over_gate = []
        self.gain_fun = gain_fun
        self.over_fun = over_fun
        self.both_gate = both_gate
        self._T_combine = T_combine  # 10
        self._T_middle = T_middle
        
        if self.both_gate:
            for i in range(0, self._T):
                # each layer obtain the same overshoot function
                self.over_gate.append(self.over_fun)
                if self.gain_fun == 'combine':
                    if i > self._T_combine:
                        self.gain_gate.append('inv')
                        #self.over_gate.append('none')
                    else:
                        self.gain_gate.append('relu') ##2
                        #self.over_gate.append(self.over_fun)
                else:
                    self.gain_gate.append(self.gain_fun)
        else:
            for i in range(0, self._T):
                if i > self._T_middle:
                    self.gain_gate.append(self.gain_fun)
                    self.over_gate.append('none')
                else:
                    self.gain_gate.append('none')
                    self.over_gate.append(self.over_fun)
                    
        print(self.gain_gate)
        print(self.over_gate)
        
        self._untied = untied
        self._coord = coord
        self._scope = scope
            
        """ Set up layers."""
        self.setup_layers()
        
        
    def setup_layers(self):
        
        Bs_     = []
        Ws_     = []
        thetas_ = []
        D_       = []
        D_over   = []
        W_g_     = []
        B_g_     = []
        b_g_     = []
        log_epsilon_ = []
        alti_    = []
        alti_over = []
        
        B = (np.transpose(self._A) / self._scale).astype(np.float32) 
        # W = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)
        B_g = (np.transpose(self._A) / self._scale).astype(np.float32)
        W_g = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)  # I - A^T * A
        b_g = np.zeros((self._N,1),dtype=np.float32)
        
        # (Old Version) D is equal to v_t in gain functions, and is set to a vector
        # To be consistant with paper, we change it as a scalar.
        # D = np.ones((self._N,1),dtype=np.float32)
        D = 1.0
        
        with tf.variable_scope (self._scope, reuse=False) as vs:
            
            # constant
            self._kPhi_ = tf.constant (value=self._Phi, dtype=tf.float32)
            self._kD_   = tf.constant (value=self._D, dtype=tf.float32)
            self._kA_   = tf.constant (value=self._A, dtype=tf.float32)
            self._vD_   = tf.get_variable (name='D', dtype=tf.float32,
                                           initializer=self._D)
            
            if not self._untied: # tied model
                Ws_.append(tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=B))
                Ws_ = Ws_ * self._T
                
            for t in range (self._T):
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                alti_.append (tf.get_variable (name="alti_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))
                alti_over.append (tf.get_variable (name="alti_over_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))

                if t < 7:
                    _logep = self._logep  # -2.0
                elif t <= 10:
                    _logep = self._logep - 2.0  # -4.0
                else:
                    _logeq = -7.0
                    
                log_epsilon_.append (tf.get_variable(name='log_epsilon_%d'%(t+1), dtype=tf.float32, initializer=_logep))
                
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name="W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=B))
                    
            for t in range(self._T):
                D_.append(tf.get_variable(name='D_%d'%(t+1), dtype=tf.float32, initializer=D))
                # D_over.append(tf.get_variable(name='D_over_%d'%(t+1), dtype=tf.float32, initializer=D))

            # if False:
            #     D_.append(tf.get_variable(name='D',dtype=tf.float32,initializer=D))
            #     D_ = D_ * self._T
                
            B_g_.append(tf.get_variable(name='B_g',shape = B_g.shape, dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer()))
            B_g_ = B_g_ * self._T

            # b_g_.append(tf.get_variable(name='b_g',shape = b_g.shape, dtype=tf.float32,
            #                             initializer=tf.glorot_uniform_initializer()))
            # b_g_ = b_g_ * self._T

            W_g_.append(tf.get_variable(name='W_g',shape = W_g.shape ,dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer()))
            W_g_ = W_g_ * self._T


        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (log_epsilon_[:-1], Ws_[:-1], thetas_[:-1], B_g_[:-1], W_g_[:-1], D_[:-1], alti_[:-1], alti_over[:-1]))
        self.vars_in_layer.append ((log_epsilon_[-1], Ws_[-1], thetas_[-1], B_g_[-1], W_g_[-1], D_[-1], alti_[-1], alti_over[-1], self._vD_, ))


    def inference(self, y_, x0_=None):
        def reweight_function(x, D, theta, alti_):
            # gain_func: piece-wise linear function
            reweight = 1.0 + alti_ * theta * tf.nn.relu(1 - tf.nn.relu(D * tf.abs(x)))
            return reweight

        def reweight_inverse(x, D, theta, alti):
            # gain_func: inverse proportional function
            reweight = 1.0 + alti * theta * 0.2 / (0.001 + D * tf.abs(x))
            return reweight

        def reweight_exp(x, D, theta, alti):
            # gain_func: exponential function
            reweight = 1.0 + alti * theta * tf.exp(- D * tf.abs(x))
            return reweight

        def reweight_sigmoid(x, D, theta, alti):
            # gain_func: not listed in paper
            reweight = 1.0 + alti * theta * tf.nn.sigmoid(- D * tf.abs(x))
            return reweight

        def reweight_inverse_variant(x, D, theta, alti, epsilon):
            # gain_func: not listed in paper
            reweight = 1.0 + alti * theta * 0.2 / (epsilon + D * tf.abs(x))
            return reweight

        def gain(x, D, theta, alti_, epsilon, gain_fun):
            if gain_fun == 'relu':
                return reweight_function(x, D, theta, alti_) + 0.0 * epsilon
            elif gain_fun == 'inv':
                return reweight_inverse(x, D, theta, alti_) + 0.0 * epsilon
            elif gain_fun == 'exp':
                return reweight_exp(x, D, theta, alti_) + 0.0 * epsilon
            elif gain_fun == 'sigm':
                return reweight_sigmoid(x, D, theta, alti_) + 0.0 * epsilon
            elif gain_fun == 'inv_v':
                return reweight_inverse_variant(x, D, theta, alti_, epsilon)
            elif gain_fun == 'none':
                return 1.0 + 0.0 * (epsilon + D + theta + alti_) 
        
        def overshoot(alti, Part_1, Part_2):
            if self._overshoot:
                return 1.0 + alti * Part_1 * Part_2
            else:
                return 1.0 
            
        xhs_  = [] # collection of the regressed sparse codes
        fhs_  = [] # collection of the regressed signals
        
        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
            
        xhs_.append(xh_)
        fhs_.append (tf.matmul (self._kD_, xh_))
        
        with tf.variable_scope (self._scope, reuse=True) as vs:
            
            for t in range (self._T):
                if t < self._T - 1:
                    log_epsilon, W_, theta_, B_g, W_g, D, alti, alti_over = self.vars_in_layer[t]
                    D_ = self._kD_
                else:
                    log_epsilon, W_, theta_, B_g, W_g, D, alti, alti_over, D_ = self.vars_in_layer [t]
                    
                By_ = tf.matmul (W_, y_)
                # By_ = tf.reduce_sum(y_)
                # Part_1_sig = tf.nn.sigmoid(tf.matmul(W_g, xh_) + tf.matmul(B_g, y_) + b_g)
                Part_1_sig = tf.nn.sigmoid(tf.matmul(W_g, xh_) + tf.matmul(B_g, y_))
                Part_2_sig = tf.abs(By_)
                
                in_ = gain(xh_, D, 1.0, alti, tf.exp(log_epsilon), self.gain_gate[t])
                Part_2_inv = theta_
                    
                res_ = y_ - tf.matmul (self._kA_, in_ * xh_)

                xh_title = shrink_free(in_ * xh_ + tf.matmul(W_, res_), theta_)
                Part_1_inv = 1.0 / (abs(xh_title - xh_) + 0.1) 
                
                if self.over_gate[t] == 'inv':
                    g_ = overshoot(alti_over, Part_1_inv, Part_2_inv)
                elif self.over_gate[t] == 'sigm':
                    g_ = overshoot(alti_over, Part_1_sig, Part_2_sig)
                elif self.over_gate[t] == 'none':
                    g_ = 1.0
                    
                xh_ = g_ * xh_title + (1 - g_) * xh_
                xhs_.append(xh_)
                
                fhs_.append (tf.matmul (D_, xh_))
                
        return xhs_, fhs_