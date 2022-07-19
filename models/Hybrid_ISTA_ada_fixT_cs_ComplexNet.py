#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink_free
from models.LISTA_base import LISTA_base

class Hybrid_ISTA_ada_fixT_cs_ComplexNet (LISTA_base):

    def __init__ (self, Phi, D, T, lam, untied, coord, scope, mode='S', CN_mode='DesNet', BN_flag = True, alpha_initial=0.0):
        """
        :CN_mode: Decide the architectures of complex networks.
                  ['DesNet','UNet','Transformer','Fc']
        """
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
        self._CN   = CN_mode
        self._BN_flag = BN_flag

        # self.conv_num = conv_num
        # self.kernel_size = kernel_size
        # self.feature_map = feature_map
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

        if self._CN == 'DesNet':

            with tf.variable_scope (self._scope, reuse=False) as vs:

                if not self._untied: # tied model
                    print('This model needs to be untied!')
                    raise ValueError

                if self._untied: # untied model
                    for j in range(self._T):
                        paras_.append(tf.get_variable(name='deconv_' + str(j + 1),
                                                      shape=[3, 3, 16, 1], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_1',
                                                        shape=[3, 3, 16, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_2',
                                                        shape=[3, 3, 16, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_3',
                                                        shape=[3, 3, 16, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='conv_' + str(j + 1) + 'b_4',
                                                      shape=[1, 1, 16, 1], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_total_.append(paras_)
                        paras_ = []
                    self.paras_ = paras_total_

        elif self._CN == 'UNet':

            with tf.variable_scope (self._scope, reuse=False) as vs:

                if not self._untied: # tied model
                    print('This model needs to be untied!')
                    raise ValueError

                if self._untied: # untied model
                    for j in range(self._T):
                        paras_.append(tf.get_variable(name='deconv_'+str(j+1)+'b_1',
                                                      shape=[3, 3, 8, 1], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_1',
                                                        shape=[3, 3, 8, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_2',
                                                        shape=[3, 3, 16, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='conv_'+str(j+1)+'b_3',
                                                        shape=[3, 3, 16, 16], dtype=tf.float32,
                                                        initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='conv_' + str(j + 1) + 'b_4',
                                                      shape=[3, 3, 16, 16], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='conv_' + str(j + 1) + 'b_5',
                                                      shape=[3, 3, 8, 8], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='deconv_' + str(j + 1) + 'b_2',
                                                      shape=[3, 3, 16, 16], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='deconv_' + str(j + 1) + 'b_3',
                                                      shape=[3, 3, 8, 16], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_.append(tf.get_variable(name='conv_' + str(j + 1) + 'b_6',
                                                      shape=[3, 3, 8, 1], dtype=tf.float32,
                                                      initializer=tf.orthogonal_initializer()))
                        paras_total_.append(paras_)
                        paras_ = []
                    self.paras_ = paras_total_

        elif self._CN == 'Transformer':

            with tf.variable_scope (self._scope, reuse=False) as vs:

                if not self._untied: # tied model
                    print('This model needs to be untied!')
                    raise ValueError

                if self._untied: # untied model
                    for j in range(self._T):
                        # paras_.append (tf.get_variable (name='patch_proj_'+str(j), 
                        #                             shape=[16, 64], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='patch_proj_'+str(j), 
                                                    shape=[1, 16, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='class_emb_'+str(j), 
                                                    shape=[1, 1, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='pos_emb_'+str(j), 
                                                    shape=[1, 17, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='query_'+str(j), 
                                                    shape=[1, 64, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='key_'+str(j), 
                                                    shape=[1, 64, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='value_'+str(j), 
                                                    shape=[1, 64, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='combine_'+str(j), 
                                                    shape=[1, 64, 64], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='mlp1_'+str(j), 
                                                    shape=[1, 64, 128], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='mlp2_'+str(j), 
                                                    shape=[1, 128, 16], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='final_fc_'+str(j), 
                                                    shape=[1, 17, 16], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='query_'+str(j), 
                        #                             shape=[64, 64], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='key_'+str(j), 
                        #                             shape=[64, 64], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='value_'+str(j), 
                        #                             shape=[64, 64], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='combine_'+str(j), 
                        #                             shape=[64, 64], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='mlp1_'+str(j), 
                        #                             shape=[64, 128], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='mlp2_'+str(j), 
                        #                             shape=[128, 16], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        # paras_.append (tf.get_variable (name='final_fc_'+str(j), 
                        #                             shape=[17, 16], dtype=tf.float32,
                        #                             initializer=tf.orthogonal_initializer()))
                        paras_total_.append(paras_)
                        paras_ = []
                        
                    self.paras_ = paras_total_
                        
        elif self._CN == 'Fc':

            with tf.variable_scope (self._scope, reuse=False) as vs:

                if not self._untied: # tied model
                    print('This model needs to be untied!')
                    raise ValueError

                if self._untied: # untied model
                    for j in range(self._T):

                        paras_.append (tf.get_variable (name='fc_'+str(j), 
                                                    shape=[16*16, 16*16], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_.append (tf.get_variable (name='dotmat_'+str(j), 
                                                    shape=[1, 16*16], dtype=tf.float32,
                                                    initializer=tf.orthogonal_initializer()))
                        paras_total_.append(paras_)
                        paras_ = []

                    self.paras_ = paras_total_
        else:
            print('No such name of complex network mode.')
            raise ValueError


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
                        
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (delta_total_[:-1], D_inv_list[:-1], ts_scalar_ [:-1], self.paras_ [:-1], self.alphas_raw [:-1], lams_[:-1]))
        self.vars_in_layer.append ((delta_total_[-1], D_inv_list[-1], ts_scalar_ [-1], self.paras_ [-1], self.alphas_raw [-1], lams_[-1], self._vD_, ))
        self.ts_ = ts_scalar_

    def lam_range(self, lam_input, x_n, x_n_1, lam_old, C=1):
        Q = C * tf.norm(x_n-x_n_1)
        P = tf.cond(Q < lam_old, lambda: Q, lambda: lam_old)
        lam_final = tf.sigmoid(lam_input) * P * 1.89    # 1.89 for lam_init = 0.1
        return lam_final

    def inference (self, y_, x0_=None):
        
        def gelu(input_tensor):
            cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
            out = input_tensor * cdf
            return out
        
        def extract_patches(images):
            patch_size = 4
            batch_size = tf.shape(images)[0]
            patches = tf.extract_image_patches(
                images=images,
                ksizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patches = tf.reshape(patches, [batch_size, -1, patch_size**2])
            return patches 
        
        
        def MultiHeadSelfAttention(x, query, key, value, combine) :
            batch_size = tf.shape(x)[0]
            
            query_out = tf.matmul(x, query)
            key_out = tf.matmul(x, key)
            value_out = tf.matmul(x, value)
            
            query_sp = tf.transpose(tf.reshape(query_out, (batch_size, -1, 4, 16)), perm=[0, 2, 1, 3])
            key_sp = tf.transpose(tf.reshape(key_out, (batch_size, -1, 4, 16)), perm=[0, 2, 1, 3])
            value_sp = tf.transpose(tf.reshape(value_out, (batch_size, -1, 4, 16)), perm=[0, 2, 1, 3])
            
            score = tf.matmul(query_sp, key_sp, transpose_b=True)
            dim_key = tf.cast(tf.shape(key_sp)[-1], tf.float32)
            scaled_score = score / tf.math.sqrt(dim_key)
            weights = tf.nn.softmax(scaled_score, axis=-1)
            attention = tf.matmul(weights, value_sp)

            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(attention, (batch_size, -1, 64))
            output = tf.matmul(concat_attention, combine)
            return output
        
        
        def Transformer_Block(inp, query, key, value, combine, mlp_1, mlp_2):
            inputs_norm = tf.contrib.layers.layer_norm(inputs=inp, center=False, 
                                                       scale=False, trainable=False,)
            # inputs_norm = LayerNormalization(epsilon=1e-6)(inputs)
            attn_output = MultiHeadSelfAttention(inputs_norm, query, key, value, combine)
            # attn_output = Dropout(0.1)(attn_output, training=self._BN_flag)
            attn_output = tf.layers.dropout(attn_output, rate=0.1, training=self._BN_flag)
            out1 = attn_output + inp

            out1_norm = tf.contrib.layers.layer_norm(inputs=out1, center=False, 
                                                       scale=False, trainable=False,)
            # out1_norm = LayerNormalization(epsilon=1e-6)(out1)
            # mlp_output = self.mlp(out1_norm)
            mlp_output_1 = gelu(tf.matmul(out1_norm, mlp_1))
            # mlp_output_2 = Dropout(0.1)(mlp_output_1, training=self._BN_flag)
            mlp_output_2 = tf.layers.dropout(mlp_output_1, rate=0.1, training=self._BN_flag)
            mlp_output_3 = tf.matmul(mlp_output_2, mlp_2)
            # mlp_output = Dropout(0.1)(mlp_output_3, training=self._BN_flag)
            mlp_output = tf.layers.dropout(mlp_output_3, rate=0.1, training=self._BN_flag)
            return mlp_output
        
        
        def Vision_Transformer_one_block(x, patch_proj, class_emb, pos_emb, 
                                         query, key, value, combine, mlp_1, mlp_2, final_fc):
            batch_size = tf.shape(x)[0]
            patches = extract_patches(x)
            x = tf.matmul(patches, patch_proj)
            class_emb = tf.broadcast_to(class_emb, [batch_size, 1, 64])
            x = tf.concat([class_emb, x], axis=1)
            x = x + pos_emb
            out = Transformer_Block(x, query, key, value, combine, mlp_1, mlp_2)
            out = tf.matmul(tf.transpose(out, [0, 2, 1]), final_fc)
            return out
        
        
        def relu_batch_normalization(x, offset, scale, res_connect_to_x=0.0, relu_flag=True):
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            output = tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.001)
            if relu_flag:
                output = tf.nn.relu(output + res_connect_to_x)
            else:
                output = output + res_connect_to_x
            return output
        
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
            for t in range (self._T):

                if t < self._T - 1:
                    del_, D_inv_, t_, dnn_para_, alpha_raw, lam_ = self.vars_in_layer [t]
                    D_ = self._kD_
                else:
                    del_, D_inv_, t_, dnn_para_, alpha_raw, lam_, D_ = self.vars_in_layer [t]

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

                # u_n
                if self._CN == 'DesNet':
                    # vh_0 = tf.expand_dims(tf.transpose(vh_), -1)

                    vh_0 = tf.reshape(tf.transpose(tf.matmul (D_, vh_)), [-1, 16, 16, 1])
                    batchsize = tf.shape(vh_0)[0]
                    vh_1 = tf.nn.conv2d_transpose(value=vh_0, filter=dnn_para_[0], output_shape=[batchsize, 32, 32, 16], strides=[1,2,2,1], padding='SAME')
                    vh_1 = relu_batch_normalization(vh_1, 0.0, 1.0)
                    # vh_1 = tf.nn.relu(tf.layers.batch_normalization(vh_1, self._BN_flag))

                    vh_2 = tf.nn.conv2d(vh_1, dnn_para_[1], strides=[1,1,1,1], padding='SAME')
                    vh_2 = relu_batch_normalization(vh_2, 0.0, 1.0, vh_1)
                    # vh_2 = tf.nn.relu(tf.layers.batch_normalization(vh_2, self._BN_flag)+ vh_1)

                    vh_3 = tf.nn.conv2d(vh_2, dnn_para_[2], strides=[1,1,1,1], padding='SAME')
                    vh_3 = relu_batch_normalization(vh_3, 0.0, 1.0, vh_1+vh_2)
                    # vh_3 = tf.nn.relu(tf.layers.batch_normalization(vh_3, self._BN_flag)+ vh_1 + vh_2)

                    vh_4 = tf.nn.conv2d(vh_3, dnn_para_[3], strides=[1,1,1,1], padding='SAME')
                    vh_4 = relu_batch_normalization(vh_4, 0.0, 1.0, vh_1+vh_2+vh_3, False)
                    # vh_4 = tf.layers.batch_normalization(vh_4, self._BN_flag+ vh_1 + vh_2 + vh_3)

                    vh_5 = tf.layers.average_pooling2d(vh_4, 2, 2)
                    vh_6 = tf.nn.conv2d(vh_5, dnn_para_[4], strides=[1,1,1,1], padding='SAME')

                    # uh_ = tf.reshape(vh_6, tf.shape(vh_)) + vh_
                    uh_ = tf.matmul(D_inv_, tf.transpose(tf.reshape(vh_6, [-1, 16*16]))) + vh_

                elif self._CN == 'UNet':
                    vh_0 = tf.reshape(tf.transpose(tf.matmul (D_, vh_)), [-1, 16, 16, 1])
                    batchsize = tf.shape(vh_0)[0]
                    vh_1 = tf.nn.conv2d_transpose(value=vh_0, filter=dnn_para_[0], output_shape=[batchsize, 32, 32, 8], strides=[1,2,2,1], padding='SAME')
                    vh_1 = relu_batch_normalization(vh_1, 0.0, 1.0)
                    # vh_1 = tf.nn.relu(tf.layers.batch_normalization(vh_1, self._BN_flag))

                    vh_2 = tf.nn.conv2d(vh_1, dnn_para_[1], strides=[1,2,2,1], padding='SAME')
                    vh_2 = relu_batch_normalization(vh_2, 0.0, 1.0)
                    # vh_2 = tf.nn.relu(tf.layers.batch_normalization(vh_2, self._BN_flag))

                    vh_3 = tf.nn.conv2d(vh_2, dnn_para_[2], strides=[1,2,2,1], padding='SAME')
                    vh_3 = relu_batch_normalization(vh_3, 0.0, 1.0)
                    # vh_3 = tf.nn.relu(tf.layers.batch_normalization(vh_3, self._BN_flag))

                    vh_4 = tf.nn.conv2d(vh_3, dnn_para_[3], strides=[1,1,1,1], padding='SAME')
                    vh_4 = relu_batch_normalization(vh_4, 0.0, 1.0)
                    # vh_4 = tf.nn.relu(tf.layers.batch_normalization(vh_4, self._BN_flag))

                    vh_5 = tf.nn.conv2d(vh_2, dnn_para_[4], strides=[1,1,1,1], padding='SAME')
                    vh_5 = relu_batch_normalization(vh_5, 0.0, 1.0)
                    # vh_5 = tf.nn.relu(tf.layers.batch_normalization(vh_5, self._BN_flag))

                    vh_6 = tf.nn.conv2d(vh_1, dnn_para_[5], strides=[1,1,1,1], padding='SAME')
                    vh_6 = relu_batch_normalization(vh_6, 0.0, 1.0)
                    # vh_6 = tf.nn.relu(tf.layers.batch_normalization(vh_6, self._BN_flag))

                    vh_7 = tf.nn.conv2d_transpose(value=vh_4, filter=dnn_para_[6], output_shape=[batchsize, 16, 16, 16], strides=[1,2,2,1], padding='SAME')
                    vh_7 = relu_batch_normalization(vh_7, 0.0, 1.0)
                    # vh_7 = tf.nn.relu(tf.layers.batch_normalization(vh_7, self._BN_flag))

                    vh_8 = tf.nn.conv2d_transpose(value=vh_5+vh_7, filter=dnn_para_[7], output_shape=[batchsize, 32, 32, 8], strides=[1,2,2,1], padding='SAME')
                    vh_8 = relu_batch_normalization(vh_8, 0.0, 1.0)
                    # vh_8 = tf.nn.relu(tf.layers.batch_normalization(vh_8, self._BN_flag))

                    vh_9 = tf.nn.conv2d(vh_6+vh_8, dnn_para_[8], strides=[1,2,2,1], padding='SAME')

                    uh_ = tf.matmul(D_inv_, tf.transpose(tf.reshape(vh_9, [-1, 16*16]))) + vh_


                elif self._CN == 'Transformer':
                    
                    vh_0 = tf.reshape(tf.transpose(tf.matmul (D_, vh_)), [-1, 16, 16, 1])
                    vh_1 = Vision_Transformer_one_block(vh_0, dnn_para_[0], dnn_para_[1], dnn_para_[2], 
                                        dnn_para_[3], dnn_para_[4], dnn_para_[5], dnn_para_[6], dnn_para_[7], dnn_para_[8], dnn_para_[9])
                    uh_ = tf.matmul(D_inv_, tf.transpose(tf.reshape(vh_1, [-1, 16*16]))) + vh_
                    
                elif self._CN == 'Fc':
                    
                    vh_0 = tf.reshape(tf.transpose(tf.matmul (D_, vh_)), [-1, 16*16])
                    vh_1 = tf.matmul(vh_0, dnn_para_[0])
                    vh_2 = gelu(vh_1)
                    vh_3 = tf.layers.dropout(vh_2, rate=0.1, training=self._BN_flag)
                    vh_4 = tf.multiply(vh_3, dnn_para_[1])
                    uh_ = tf.matmul(D_inv_, tf.transpose(tf.reshape(vh_4, [-1, 16*16]))) + vh_
  
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

        return xhs_, fhs_
