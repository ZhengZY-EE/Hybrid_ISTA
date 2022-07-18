'''
Description: Hybrid GLISTA
Version: 1.0
Autor: https://github.com/wukailun/GLISTA/blob/master/GLISTA_cp.py
Date: 2021-11-15 14:18:40
LastEditors: Ziyang Zheng
LastEditTime: 2022-01-28 02:29:42
'''
import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base


### This in the main coding of the GLISTA with couple parameters
class Hybrid_GLISTA_cs (LISTA_base):
    
    """
    Implementation of hybrid GLISTA_cs.
    """
    def __init__(self, Phi, D, T, lam, untied, coord, scope, alti, overshoot, gain_fun, over_fun, both_gate, T_combine, T_middle, conv_num=3, kernel_size=3, feature_map=16, alpha_initial=0.0):
        """
        :prob:     : Instance of Problem class, describing problem settings.
        :T         : Number of layers (depth) of this LISTA model.
        :lam       : Initial value of thresholds of MT.
        :untied    : Whether weights are shared within layers.
        :alti      : Initial mu_t in gain functions.
        :overshoot : Whether we adopt overshoot gate.
        :gain_fun  : The name of gain functions, [inv, combine, none].
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
        self._T_combine = T_combine
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
        
        Ws1_     = []
        thetas1_ = []
        Ws2_     = []
        thetas2_ = []
        W_g1_     = []
        B_g1_     = []
        W_g2_     = []
        B_g2_     = []
        log_epsilon_ = []
        alti_over1 = []
        alti_over2 = []
        roi = []
        learn_vec = []
        
        D_inv_list = []
        
        B = (np.transpose(self._A) / self._scale).astype(np.float32) 
        # W = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)
        B_g = (np.transpose(self._A) / self._scale).astype(np.float32)
        W_g = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)  # I - A^T * A
        
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
            
            self._D_inv = tf.get_variable (name='D_inv',shape=[self._N, self._F],
                                           dtype=tf.float32, initializer=tf.orthogonal_initializer())
            D_inv_list.append(self._D_inv)
            D_inv_list = D_inv_list * self._T
            
            if not self._untied: # tied model
                Ws1_.append(tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=B))
                Ws1_ = Ws1_ * self._T
                Ws2_ = Ws1_
                
            for t in range (self._T):
                thetas1_.append (tf.get_variable (name="theta1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                thetas2_.append (tf.get_variable (name="theta2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                alti_over1.append (tf.get_variable (name="alti_over1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))
                alti_over2.append (tf.get_variable (name="alti_over2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))
                roi.append (tf.get_variable (name="roi_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=0.9))
                learn_vec.append (tf.get_variable (name="learn_vec_%d"%(t+1),
                                                   dtype=tf.float32, 
                                                   initializer=1.0))

                if t < 7:
                    _logep = self._logep  # -2.0
                elif t <= 10:
                    _logep = self._logep - 2.0  # -4.0
                else:
                    _logeq = -7.0
                    
                log_epsilon_.append (tf.get_variable(name='log_epsilon_%d'%(t+1), dtype=tf.float32, initializer=_logep))
                
                if self._untied: # untied model
                    Ws1_.append (tf.get_variable (name="W1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=B))
                    Ws2_.append (tf.get_variable (name="W2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=B))
                
            B_g1_.append(tf.get_variable(name='B_g1',shape = B_g.shape, dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer()))
            B_g1_ = B_g1_ * self._T

            W_g1_.append(tf.get_variable(name='W_g1',shape = W_g.shape ,dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer()))
            W_g1_ = W_g1_ * self._T
            
            B_g2_.append(tf.get_variable(name='B_g2',shape = B_g.shape, dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer()))
            B_g2_ = B_g2_ * self._T

            W_g2_.append(tf.get_variable(name='W_g2',shape = W_g.shape ,dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer()))
            W_g2_ = W_g2_ * self._T


        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (D_inv_list[:-1], log_epsilon_[:-1], Ws1_[:-1], thetas1_[:-1], Ws2_[:-1], thetas2_[:-1], B_g1_[:-1], W_g1_[:-1], B_g2_[:-1], W_g2_[:-1], alti_over1[:-1], alti_over2[:-1], roi[:-1], learn_vec[:-1], self.paras_[:-1], self.alphas_raw[:-1]))
        self.vars_in_layer.append((D_inv_list[-1], log_epsilon_[-1], Ws1_[-1], thetas1_[-1], Ws2_[-1], thetas2_[-1], B_g1_[-1], W_g1_[-1], B_g2_[-1], W_g2_[-1], alti_over1[-1], alti_over2[-1], roi[-1], learn_vec[-1], self.paras_[-1], self.alphas_raw[-1], self._vD_, ))


    def inference(self, y_, x0_=None):
        def reweight_function(roi, theta_min, vw_max, learn_vec):
            # gain_func: piece-wise linear function
            reweight = 1.0 + roi * theta_min * tf.nn.relu(1 - tf.nn.relu(learn_vec * tf.abs(vw_max)))
            return reweight
        
        def reweight_inv(roi, theta_min, vw_max, learn_vec):
            # out_part = tf.sigmoid(learn_vec * x)
            # bound_left = (1 - roi) * theta_max / (vw_min + 0.000001)
            # bound_right = (1 + roi) * theta_min / (vw_max + 0.000001)
            # reweight = tf.multiply(bound_right - bound_left, out_part) + bound_left + 1.0
            reweight = 1.0 + roi * (theta_min) * 0.2 / (0.001 + learn_vec * tf.abs(vw_max))
            return reweight

        def gain(roi, theta_min, vw_max, learn_vec, epsilon, gain_fun):
            if gain_fun == 'inv':
                return reweight_inv(roi, theta_min, vw_max, learn_vec) + 0.0 * epsilon
            elif gain_fun == 'relu':
                return reweight_function(roi, theta_min, vw_max, learn_vec) + 0.0 * epsilon
            elif gain_fun == 'none':
                return 1.0 + 0.0 * epsilon
        
        def overshoot(alti, Part_1, Part_2):
            if self._overshoot:
                return 1.0 + alti * Part_1 * Part_2
            else:
                return 1.0 
            
        def generate_bound(theta1, theta2, v, w):
            theta_min = tf.minimum(theta1, theta2)
            # theta_max = tf.maximum(theta1, theta2)
            # vw_min = tf.minimum(tf.abs(v), tf.abs(w))
            vw_max = tf.maximum(tf.abs(v), tf.abs(w))
            return theta_min, vw_max
            
            
        xhs_ = [] # collection of the regressed sparse codesnm
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
                    D_inv_, log_epsilon, W1_, theta1_, W2_, theta2_, B_g1, W_g1, B_g2, W_g2, alti_over1, alti_over2, roi, learn_vec, dnn_para_, alpha_raw = self.vars_in_layer[t]
                    D_ = self._kD_
                else:
                    D_inv_, log_epsilon, W1_, theta1_, W2_, theta2_, B_g1, W_g1, B_g2, W_g2, alti_over1, alti_over2, roi, learn_vec, dnn_para_, alpha_raw, D_ = self.vars_in_layer[t]
                
                if t == 0:
                    
                    # v_n: [500, 64]
                    res1_ = y_ - tf.matmul (self._kA_, xh_)
                    vh_ = shrink_free (xh_ + tf.matmul (W1_, res1_), tf.abs(theta1_))
                    
                    vh_ = vh_ + 0.0*log_epsilon*alti_over1*alti_over2*roi*tf.reduce_sum(B_g1+B_g2)*tf.reduce_sum(learn_vec)*tf.reduce_sum(W_g1+W_g2)

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
                    wh_ = shrink_free (uh_ + tf.matmul (W2_, res2_), tf.abs(theta2_))

                    # 1-alpha_n (0, 1 - lower bound of alpha)
                    one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(theta1_) / (tf.abs(theta1_)+tf.abs(theta2_))

                    # x_n+1
                    xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                    
                    theta1_old = theta1_
                    theta2_old = theta2_
                    
                else:
                    
                    # v_n: [500, 64]
                    theta_min, vw_max = generate_bound(theta1_old, theta2_old, vh_, wh_)
                    in_ = gain(roi, theta_min, vw_max, learn_vec, tf.exp(log_epsilon), self.gain_gate[t])  
                    res_ = y_ - tf.matmul (self._kA_, in_ * xh_)
                    xh_title = shrink_free(in_ * xh_ + tf.matmul(W1_, res_), theta1_)
                    
                    By_ = tf.matmul (W1_, y_)
                    Part_1_sig = tf.nn.sigmoid(tf.matmul(W_g1, xh_) + tf.matmul(B_g1, y_))
                    Part_2_sig = tf.abs(By_)
                    Part_1_inv = 1.0 / (abs(xh_title - xh_) + 0.1) 
                    Part_2_inv = theta1_
                    
                    if self.over_gate[t] == 'inv':
                        g_ = overshoot(alti_over1, Part_1_inv, Part_2_inv)
                    elif self.over_gate[t] == 'sigm':
                        g_ = overshoot(alti_over1, Part_1_sig, Part_2_sig)
                    elif self.over_gate[t] == 'none':
                        g_ = 1.0
                        
                    vh_ = g_ * xh_title + (1 - g_) * xh_
                    
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
                    res2_ = y_ - tf.matmul (self._kA_, in_ * uh_)
                    uh_title = shrink_free(in_ * uh_ + tf.matmul(W2_, res2_), theta2_)
                    
                    By_w = tf.matmul (W2_, y_)
                    Part_w1_sig = tf.nn.sigmoid(tf.matmul(W_g2, uh_) + tf.matmul(B_g2, y_))
                    Part_w2_sig = tf.abs(By_w)
                    Part_w1_inv = 1.0 / (abs(uh_title - uh_) + 0.1) 
                    Part_w2_inv = theta2_
                    
                    if self.over_gate[t] == 'inv':
                        g2_ = overshoot(alti_over2, Part_w1_inv, Part_w2_inv)
                    elif self.over_gate[t] == 'sigm':
                        g2_ = overshoot(alti_over2, Part_w1_sig, Part_w2_sig)
                    elif self.over_gate[t] == 'none':
                        g2_ = 1.0
                        
                    wh_ = g2_ * uh_title + (1 - g2_) * uh_
                    
                    # 1-alpha_n (0, 1 - lower bound of alpha)
                    one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(theta1_) / (tf.abs(theta1_)+tf.abs(theta2_))

                    # x_n+1
                    xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                    
                    theta1_old = theta1_
                    theta2_old = theta2_
                    
                xhs_.append(xh_)
                fhs_.append (tf.matmul (D_, xh_))
                
        return xhs_, fhs_