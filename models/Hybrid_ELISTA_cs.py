'''
Description: hybrid ELISTA.
Version: 1.0
Autor: Ziyang Zheng
Date: 2021-11-22 15:25:42
LastEditors: Ziyang Zheng
LastEditTime: 2022-01-28 02:29:53
'''
import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import MT, shrink_free
from models.LISTA_base import LISTA_base

class Hybrid_ELISTA_cs (LISTA_base):

    """
    Implementation of hybrid ELISTA_cs.
    """
    def __init__ (self, Phi, D, T, lam, untied, coord, scope, mt_flag, conv_num=3, kernel_size=3, feature_map=16, alpha_initial=0.0):
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
        Ws1_     = []
        Ws2_     = []
        alphas1_  = []
        alphas2_  = []
        thetas1_ = []
        thetas2_ = []
        
        Ws3_     = []
        Ws4_     = []
        alphas3_  = []
        alphas4_  = []
        thetas3_ = []
        thetas4_ = []

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
                Ws3_ = Ws1_
                Ws4_ = Ws1_
                
                for t in range (self._T):
                    alphas1_.append (tf.get_variable (name="alpha1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                    alphas2_.append (tf.get_variable (name="alpha2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                    alphas3_.append (tf.get_variable (name="alpha3_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                    alphas4_.append (tf.get_variable (name="alpha4_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=1.0))
            else:
                
                alphas1_.append (tf.get_variable (name="alpha",
                                                 dtype=tf.float32,
                                                 initializer=1.0))
                alphas1_ = alphas1_ * self._T
                alphas2_ = alphas1_
                alphas3_ = alphas1_
                alphas4_ = alphas1_
                
                for t in range (self._T):
                    Ws1_.append (tf.get_variable (name="W1_%d"%(t+1),
                                                        dtype=tf.float32,
                                                        initializer=W))
                    Ws2_.append (tf.get_variable (name="W2_%d"%(t+1),
                                                        dtype=tf.float32,
                                                        initializer=W))
                    Ws3_.append (tf.get_variable (name="W3_%d"%(t+1),
                                                        dtype=tf.float32,
                                                        initializer=W))
                    Ws4_.append (tf.get_variable (name="W4_%d"%(t+1),
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
                    thetas3_.append (tf.get_variable (name="theta3_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta))
                    thetas4_.append (tf.get_variable (name="theta4_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=self._theta)) 
                         
                self.vars_in_layer = list (zip (D_inv_list[:-1], Ws1_[:-1], Ws2_[:-1], Ws3_[:-1], Ws4_[:-1], alphas1_[:-1], alphas2_[:-1], alphas3_[:-1], alphas4_[:-1], thetas1_[:-1], thetas2_[:-1], thetas3_[:-1], thetas4_[:-1], self.paras_[:-1], self.alphas_raw[:-1])) 
                self.vars_in_layer.append((D_inv_list[-1], Ws1_[-1], Ws2_[-1], Ws3_[-1], Ws4_[-1], alphas1_[-1], alphas2_[-1], alphas3_[-1], alphas4_[-1], thetas1_[-1], thetas2_[-1], thetas3_[-1], thetas4_[-1], self.paras_[-1], self.alphas_raw[-1], self._vD_, )) 
                  
            else:
                thetas1_bar_ = []
                thetas2_bar_ = []
                thetas3_bar_ = []
                thetas4_bar_ = []
                for t in range (self._T):
                    thetas1_.append (tf.get_variable (name="theta1_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta).astype(np.float32)))
                    thetas2_.append (tf.get_variable (name="theta2_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta).astype(np.float32)))  
                    thetas3_.append (tf.get_variable (name="theta3_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta).astype(np.float32))) 
                    thetas4_.append (tf.get_variable (name="theta4_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta).astype(np.float32))) 
                    thetas1_bar_.append (tf.get_variable (name="theta1_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta+0.1).astype(np.float32)))
                    thetas2_bar_.append (tf.get_variable (name="theta2_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta+0.1).astype(np.float32)))  
                    thetas3_bar_.append (tf.get_variable (name="theta3_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta+0.1).astype(np.float32)))  
                    thetas4_bar_.append (tf.get_variable (name="theta4_bar_%d"%(t+1),
                                                    dtype=tf.float32,
                                                    initializer=(self._theta+0.1).astype(np.float32)))  
                
                self.vars_in_layer = list (zip (D_inv_list[:-1], Ws1_[:-1], Ws2_[:-1], Ws3_[:-1], Ws4_[:-1], alphas1_[:-1], alphas2_[:-1], alphas3_[:-1], alphas4_[:-1], thetas1_[:-1], thetas2_[:-1], thetas3_[:-1], thetas4_[:-1], thetas1_bar_[:-1], thetas2_bar_[:-1], thetas3_bar_[:-1], thetas4_bar_[:-1], self.paras_[:-1], self.alphas_raw[:-1])) 
                self.vars_in_layer.append((D_inv_list[-1], Ws1_[-1], Ws2_[-1], Ws3_[-1], Ws4_[-1], alphas1_[-1], alphas2_[-1], alphas3_[-1], alphas4_[-1], thetas1_[-1], thetas2_[-1], thetas3_[-1], thetas4_[-1], thetas1_bar_[-1], thetas2_bar_[-1], thetas3_bar_[-1], thetas4_bar_[-1], self.paras_[-1], self.alphas_raw[-1], self._vD_, ))

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
                        D_inv_, W1_, W2_, W3_, W4_, alpha1_, alpha2_, alpha3_, alpha4_, theta1_, theta2_, theta3_, theta4_, dnn_para_, alpha_raw = self.vars_in_layer [t]
                        D_ = self._kD_
                    else:
                        D_inv_, W1_, W2_, W3_, W4_, alpha1_, alpha2_, alpha3_, alpha4_, theta1_, theta2_, theta3_, theta4_, dnn_para_, alpha_raw, D_ = self.vars_in_layer [t]
                    
                    alpha1_ = tf.sigmoid(alpha1_)

                    # v_n: [500, 64]
                    res1_ = y_ - tf.matmul (self._kA_, xh_)
                    vh_mid_ = shrink_free (xh_ + alpha1_ * tf.matmul (W1_, res1_), theta1_)
                    res2_ = y_ - tf.matmul (self._kA_, vh_mid_)
                    vh_ = shrink_free (xh_ + alpha2_ * tf.matmul (W2_, res2_), theta2_)
                    
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
                    res3_ = y_ - tf.matmul (self._kA_, uh_)
                    wh_mid_ = shrink_free (uh_ + alpha3_ * tf.matmul (W3_, res3_), theta3_)
                    res4_ = y_ - tf.matmul (self._kA_, wh_mid_)
                    wh_ = shrink_free (uh_ + alpha4_ * tf.matmul (W4_, res4_), theta4_)
                    
                    # 1-alpha_n (0, 1 - lower bound of alpha)
                    one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(alpha2_ * theta1_+theta2_) / (tf.abs(alpha2_ * theta1_+theta2_)+tf.abs(alpha4_ * theta3_+theta4_))

                    # x_n+1
                    xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                    xhs_.append (xh_)
                    fhs_.append (tf.matmul (D_, xh_))
            else:
                for t in range (self._T):
                    if t < self._T - 1:
                        D_inv_, W1_, W2_, W3_, W4_, alpha1_, alpha2_, alpha3_, alpha4_, theta1_, theta2_, theta3_, theta4_, theta1_bar_, theta2_bar_, theta3_bar_, theta4_bar_, dnn_para_, alpha_raw = self.vars_in_layer [t]
                        D_ = self._kD_
                    else:
                        D_inv_, W1_, W2_, W3_, W4_, alpha1_, alpha2_, alpha3_, alpha4_, theta1_, theta2_, theta3_, theta4_, theta1_bar_, theta2_bar_, theta3_bar_, theta4_bar_, dnn_para_, alpha_raw, D_ = self.vars_in_layer [t]
                    
                    alpha1_ = tf.sigmoid(alpha1_)

                    # v_n: [500, 64]
                    res1_ = y_ - tf.matmul (self._kA_, xh_)
                    vh_mid_ = MT (xh_ + alpha1_ * tf.matmul (W1_, res1_), theta1_, theta1_bar_)
                    res2_ = y_ - tf.matmul (self._kA_, vh_mid_)
                    vh_ = MT (xh_ + alpha2_ * tf.matmul (W2_, res2_), theta2_, theta2_bar_)
                    
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
                    alpha3_ = tf.abs(alpha3_) + 1.0
                    alpha4_ = tf.sigmoid(alpha4_)
                    res3_ = y_ - tf.matmul (self._kA_, uh_)
                    wh_mid_ = MT (uh_ + alpha3_ * tf.matmul (W3_, res3_), theta3_, theta3_bar_)
                    res4_ = y_ - tf.matmul (self._kA_, wh_mid_)
                    wh_ = MT (uh_ + alpha4_ * tf.matmul (W4_, res4_), theta4_, theta4_bar_)
                    
                    # 1-alpha_n (0, 1 - lower bound of alpha)
                    one_alpha_ = tf.sigmoid(alpha_raw) * tf.abs(alpha2_ * theta1_+theta2_) / (tf.abs(alpha2_ * theta1_+theta2_)+tf.abs(alpha4_ * theta3_+theta4_))

                    # x_n+1
                    xh_ = one_alpha_ * wh_ + (1 - one_alpha_) * vh_
                    xhs_.append (xh_)
                    fhs_.append (tf.matmul (D_, xh_))
        return xhs_, fhs_