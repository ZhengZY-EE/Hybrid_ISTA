#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : main.py
author: Xiaohan Chen
email : chernxh@tamu.edu
last_modified: 2018-10-13

Main script. Start running model from main.py.
"""

import os , sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

# timing
import time
from datetime import timedelta

from config import get_config
import utils.prob as problem
import utils.data as data
import utils.train as train
from imageio import imwrite 

import numpy as np
import tensorflow as tf
try :
    from sklearn.feature_extraction.image \
            import extract_patches_2d, reconstruct_from_patches_2d
except Exception as e :
    pass


def setup_model(config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf = 'c' if config.coord  else 's'
    mt_flag = 'mt' if config.mt_flag  else 'st'
    overshoot = 'Ogate' if config.overshoot  else 'No_Ogate'
    both_gate = 'BG' if config.both_gate else 'No_BG'
    
    if config.net == 'ISTA':
        """ISTA"""
        config.model = ("ISTA_T{T}_lam{lam}"
                        .format (T=config.T, lam=config.lam))
        from models.ISTA import ISTA
        model = ISTA (kwargs['A'], T=config.T, lam=config.lam)
        
    if config.net == 'FISTA':
        """FISTA"""
        config.model = ("FISTA_T{T}_lam{lam}"
                        .format (T=config.T, lam=config.lam))
        from models.FISTA import FISTA
        model = FISTA (kwargs['A'], T=config.T, lam=config.lam)
        
    # if config.net == 'ADMM':
    #     """ADMM"""
    #     config.model = ("ADMM_T{T}_lam{lam}"
    #                     .format (T=config.T, lam=config.lam))
    #     from models.ADMM import ADMM
    #     model = ADMM (kwargs['A'], T=config.T, lam=config.lam)

    if config.net == 'LISTA' :
        """LISTA"""
        config.model = ("LISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA import LISTA
        model = LISTA (kwargs['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)

    if config.net == 'LAMP' :
        """LAMP"""
        config.model = ("LAMP_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LAMP import LAMP
        model = LAMP (kwargs['A'], T=config.T, lam=config.lam,
                      untied=config.untied, coord=config.coord,
                      scope=config.scope)

    # if config.net == 'LIHT' :
    #     """LIHT"""
    #     from models.LIHT import LIHT
    #     model = LIHT (p, T=config.T, lam=config.lam, y_=p.y_ , x0_=None ,
    #                   untied=config.untied , cord=config.coord)
    
    if config.net == 'ELISTA' :
        """ELISTA"""
        config.model = ("ELISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{mt_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, mt_flag=mt_flag, exp_id=config.exp_id))
        from models.ELISTA import ELISTA
        model = ELISTA (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord, mt_flag=config.mt_flag,
                          scope=config.scope)
        
    if config.net == 'ELISTA_cs' :
        """ELISTA_cs"""
        config.model = ("ELISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{mt_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf, mt_flag=mt_flag, 
                                 exp_id=config.exp_id))
        from models.ELISTA_cs import ELISTA_cs
        model = ELISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                            lam=config.lam,
                            untied=config.untied, coord=config.coord, mt_flag=config.mt_flag,
                            scope=config.scope)
        
    if config.net == 'Hybrid_ELISTA' :
        """Hybrid_ELISTA"""
        config.model = ("Hybrid_ELISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{mt_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, mt_flag=mt_flag, exp_id=config.exp_id))
        from models.Hybrid_ELISTA import Hybrid_ELISTA
        model = Hybrid_ELISTA (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord, mt_flag=config.mt_flag,
                          scope=config.scope, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_ELISTA_cs' :
        """Hybrid_ELISTA_cs"""
        config.model = ("Hybrid_ELISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{mt_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf, mt_flag=mt_flag, 
                                 exp_id=config.exp_id))
        from models.Hybrid_ELISTA_cs import Hybrid_ELISTA_cs
        model = Hybrid_ELISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                            lam=config.lam,
                            untied=config.untied, coord=config.coord, mt_flag=config.mt_flag,
                            scope=config.scope, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)    
    
    if config.net == 'GLISTA' :
        """GLISTA"""
        config.model = ("GLISTA_T{T}_lam{lam}_{untiedf}_{coordf}_G{gain_fun}_{overshoot}_O{over_fun}_{both_gate}_{T_combine}_{T_middle}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, gain_fun=config.gain_fun, overshoot=overshoot, 
                                 over_fun=config.over_fun, both_gate=both_gate, 
                                 T_combine=config.T_combine, T_middle=config.T_middle, exp_id=config.exp_id))
        from models.GLISTA import GLISTA
        model = GLISTA (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, alti=config.alti, overshoot=config.overshoot, 
                          gain_fun=config.gain_fun, over_fun=config.over_fun, both_gate=config.both_gate, 
                          T_combine=config.T_combine, T_middle=config.T_middle)
        
    if config.net == 'GLISTA_cs' :
        """GLISTA_cs"""
        config.model = ("GLISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_G{gain_fun}_{overshoot}_O{over_fun}_{both_gate}_{T_combine}_{T_middle}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf, gain_fun=config.gain_fun, overshoot=overshoot, 
                                 over_fun=config.over_fun, both_gate=both_gate, 
                                 T_combine=config.T_combine, T_middle=config.T_middle, 
                                 exp_id=config.exp_id))
        from models.GLISTA_cs import GLISTA_cs
        model = GLISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                            lam=config.lam,
                            untied=config.untied, coord=config.coord,
                            scope=config.scope, alti=config.alti, overshoot=config.overshoot, 
                          gain_fun=config.gain_fun, over_fun=config.over_fun, both_gate=config.both_gate, 
                          T_combine=config.T_combine, T_middle=config.T_middle)
        
    if config.net == 'Hybrid_GLISTA' :
        """Hybrid_GLISTA"""
        config.model = ("Hybrid_GLISTA_T{T}_lam{lam}_{untiedf}_{coordf}_G{gain_fun}_{overshoot}_O{over_fun}_{both_gate}_{T_combine}_{T_middle}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, gain_fun=config.gain_fun, overshoot=overshoot, 
                                 over_fun=config.over_fun, both_gate=both_gate, T_combine=config.T_combine,
                                 T_middle=config.T_middle, exp_id=config.exp_id))
        from models.Hybrid_GLISTA import Hybrid_GLISTA
        model = Hybrid_GLISTA (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, alti=config.alti, overshoot=config.overshoot, 
                          gain_fun=config.gain_fun, over_fun=config.over_fun, both_gate=config.both_gate, 
                          T_combine=config.T_combine, T_middle=config.T_middle, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_GLISTA_cs' :
        """Hybrid_GLISTA_cs"""
        config.model = ("Hybrid_GLISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_G{gain_fun}_{overshoot}_O{over_fun}_{both_gate}_{T_combine}_{T_middle}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf, gain_fun=config.gain_fun, overshoot=overshoot, 
                                 over_fun=config.over_fun, both_gate=both_gate, 
                                 T_combine=config.T_combine, T_middle=config.T_middle, 
                                 exp_id=config.exp_id))
        from models.Hybrid_GLISTA_cs import Hybrid_GLISTA_cs
        model = Hybrid_GLISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                            lam=config.lam,
                            untied=config.untied, coord=config.coord,
                            scope=config.scope, alti=config.alti, overshoot=config.overshoot, 
                          gain_fun=config.gain_fun, over_fun=config.over_fun, both_gate=config.both_gate, 
                          T_combine=config.T_combine, T_middle=config.T_middle, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)

    if config.net == 'LISTA_cp' :
        """LISTA-CP"""
        config.model = ("LISTA_cp_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cp import LISTA_cp
        model = LISTA_cp (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope)

    if config.net == 'LISTA_cp_cs' :
        """LISTA-CP-CS"""
        config.model = ("LISTA_cp_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cp_cs import LISTA_cp_cs
        model = LISTA_cp_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                               lam=config.lam,
                               untied=config.untied, coord=config.coord,
                               scope=config.scope)

    if config.net == 'Hybrid_LISTA_cp' :
        """Hybrid_LISTA-CP"""
        config.model = ("Hybrid_LISTA_cp_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp import Hybrid_LISTA_cp
        model = Hybrid_LISTA_cp (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_LISTA_cp_FixNetFunc' :
        """Hybrid_LISTA-CP_FixNetFunc"""
        config.model = ("Hybrid_LISTA_cp_FixNetFunc{NetFunc}_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (NetFunc=config.net_func, T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp_FixNetFunc import Hybrid_LISTA_cp_FixNetFunc
        model = Hybrid_LISTA_cp_FixNetFunc (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode,  
                          alpha_initial=config.alpha_init, Net_Func=config.net_func)
        
    if config.net == 'Hybrid_LISTA_cp_ComplexNet':
        """Hybrid_LISTA_cp_ComplexNet"""
        config.model = ("Hybrid_LISTA_cp_ComplexNet_T{T}_lam{lam}_{untiedf}_{coordf}_w_mode{w_mode}_CN{cn_mode}_BNDrop{BN_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, w_mode = config.w_mode, cn_mode=config.CN_mode, 
                                 BN_flag=config.BN_flag, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp_ComplexNet import Hybrid_LISTA_cp_ComplexNet
        model = Hybrid_LISTA_cp_ComplexNet (kwargs['A'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, CN_mode=config.CN_mode, 
                          BN_flag=config.BN_flag, alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_LISTA_cp_ComplexNet_cs':
        """Hybrid_LISTA_cp_ComplexNet_cs"""
        config.model = ("Hybrid_LISTA_cp_ComplexNet_cs_T{T}_lam{lam}_llam{llam}_{untiedf}_{coordf}_w_mode{w_mode}_CN{cn_mode}_BNDrop{BN_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf, llam=config.lasso_lam,
                                 coordf=coordf, w_mode = config.w_mode, cn_mode=config.CN_mode, 
                                 BN_flag=config.BN_flag, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp_ComplexNet_cs import Hybrid_LISTA_cp_ComplexNet_cs
        model = Hybrid_LISTA_cp_ComplexNet_cs (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, CN_mode=config.CN_mode, 
                          BN_flag=config.BN_flag, alpha_initial=config.alpha_init)
    
    if config.net == 'Hybrid_LISTA_cp_cs' :
        """Hybrid_LISTA-CP-cs"""
        config.model = ("Hybrid_LISTA_cp_cs_T{T}_lam{lam}_llam{llam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf, llam=config.lasso_lam,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp_cs import Hybrid_LISTA_cp_cs
        model = Hybrid_LISTA_cp_cs (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)

    if config.net == 'Hybrid_LISTA_cp_cs_ISTANet_Plus' :
        """Hybrid_LISTA-CP-cs with ISTANet_Plus"""
        config.model = ("Hybrid_LISTA_cp_cs_ISTANet_Plus_T{T}_lam{lam}_llam{llam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf, llam=config.lasso_lam,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cp_cs_ISTANet_Plus import Hybrid_LISTA_cp_cs_ISTANet_Plus
        model = Hybrid_LISTA_cp_cs_ISTANet_Plus (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)     

    if config.net == 'ISTA_Net_Plus' :
        """ISTA_Net_Plus"""
        config.model = ("ISTA_Net_Plus_T{T}_{untiedf}_{exp_id}"
                        .format (T=config.T, untiedf=untiedf, exp_id=config.exp_id))
        from models.ISTA_Net_Plus import ISTA_Net_Plus
        model = ISTA_Net_Plus (kwargs['Phi'], T=config.T, untied=config.untied, 
                          scope=config.scope, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map)                  

    if config.net == 'Hybrid_ISTA_ada_fixT' :
        """Hybrid_ISTA_ada_fixT"""
        config.model = ("Hybrid_ISTA_ada_fixT_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT import Hybrid_ISTA_ada_fixT
        # in this model, lam = config.llam != config.lam
        model = Hybrid_ISTA_ada_fixT (kwargs['A'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_ISTA_ada_fixT_assump1' :
        """Hybrid_ISTA_ada_fixT_assump1"""
        config.model = ("Hybrid_ISTA_ada_fixT_assump1_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_assump1 import Hybrid_ISTA_ada_fixT_assump1
        # in this model, lam = config.llam != config.lam
        model = Hybrid_ISTA_ada_fixT_assump1 (kwargs['A'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_ISTA_ada_fixT_ComplexNet' :
        """Hybrid_ISTA_ada_fixT_ComplexNet"""
        config.model = ("Hybrid_ISTA_ada_fixT_ComplexNet_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_CN{cn_mode}_BNDrop{BN_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, cn_mode=config.CN_mode, 
                                 BN_flag=config.BN_flag, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_ComplexNet import Hybrid_ISTA_ada_fixT_ComplexNet
        # in this model, lam = config.llam != config.lam
        model = Hybrid_ISTA_ada_fixT_ComplexNet (kwargs['A'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, CN_mode=config.CN_mode, 
                          BN_flag=config.BN_flag,
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_ISTA_ada_fixT_FixNetFunc' :
        """Hybrid_ISTA_ada_fixT_FixNetFunc"""
        config.model = ("Hybrid_ISTA_ada_fixT_FixNetFunc{NetFunc}_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (NetFunc=config.net_func, T=config.T, lam=config.lasso_lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_FixNetFunc import Hybrid_ISTA_ada_fixT_FixNetFunc
        # in this model, lam = config.llam != config.lam
        model = Hybrid_ISTA_ada_fixT_FixNetFunc (kwargs['A'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, 
                          alpha_initial=config.alpha_init, Net_Func=config.net_func)
        
    if config.net == 'Hybrid_ISTA_ada_freeT' :
        """Hybrid_ISTA_ada_freeT"""
        config.model = ("Hybrid_ISTA_ada_freeT_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_freeT import Hybrid_ISTA_ada_freeT
        # in this model, lam = config.llam != config.lam
        model = Hybrid_ISTA_ada_freeT (kwargs['A'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)

    if config.net == 'Hybrid_ISTA_ada_fixT_cs' :
        """Hybrid_ISTA_ada_fixT_cs"""
        config.model = ("Hybrid_ISTA_ada_fixT_cs_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf, 
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_cs import Hybrid_ISTA_ada_fixT_cs
        model = Hybrid_ISTA_ada_fixT_cs (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_ISTA_ada_fixT_cs_ComplexNet' :
        """Hybrid_ISTA_ada_fixT_cs_ComplexNet"""
        config.model = ("Hybrid_ISTA_ada_fixT_cs_ComplexNet_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_CN{cn_mode}_BNDrop{BN_flag}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf, 
                                 coordf=coordf, w_mode=config.w_mode, cn_mode=config.CN_mode, 
                                 BN_flag=config.BN_flag, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_cs_ComplexNet import Hybrid_ISTA_ada_fixT_cs_ComplexNet
        model = Hybrid_ISTA_ada_fixT_cs_ComplexNet (kwargs['Phi'], kwargs['D'], T=config.T, 
                                                    lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, CN_mode=config.CN_mode, 
                                 BN_flag=config.BN_flag, 
                          alpha_initial=config.alpha_init)

    if config.net == 'Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus' :
        """Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus"""
        config.model = ("Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf, 
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus import Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus
        model = Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lasso_lam,
                          untied=config.untied, coord=config.coord,
                          scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                          kernel_size=config.kernel_size, feature_map=config.feature_map, 
                          alpha_initial=config.alpha_init)

    if config.net == 'Hybrid_ISTA_ada_freeT_cs' :
       """Hybrid_ISTA_ada_freeT_cs"""
       config.model = ("Hybrid_ISTA_ada_freeT_cs_T{T}_lam{lam}_{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                       .format (T=config.T, lam=config.lasso_lam, untiedf=untiedf, 
                                coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
       from models.Hybrid_ISTA_ada_freeT_cs import Hybrid_ISTA_ada_freeT_cs
       model = Hybrid_ISTA_ada_freeT_cs (kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lasso_lam,
                         untied=config.untied, coord=config.coord,
                         scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                         kernel_size=config.kernel_size, feature_map=config.feature_map, 
                         alpha_initial=config.alpha_init)

    if config.net == 'LISTA_ss' :
        """LISTA-SS"""
        config.model = ("LISTA_ss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_ss import LISTA_ss
        model = LISTA_ss (kwargs['A'], T=config.T, lam=config.lam,
                          percent=config.percent, max_percent=config.max_percent,
                          untied=config.untied , coord=config.coord,
                          scope=config.scope)

    if config.net == 'LISTA_cpss' :
        """LISTA-CPSS"""
        config.model = ("LISTA_cpss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA_cpss import LISTA_cpss
        model = LISTA_cpss (kwargs['A'], T=config.T, lam=config.lam,
                            percent=config.percent, max_percent=config.max_percent,
                            untied=config.untied , coord=config.coord,
                            scope=config.scope)
    
    if config.net == 'Hybrid_LISTA_cpss' :
        """Hybrid_LISTA-CPSS"""
        config.model = ("Hybrid_LISTA_cpss_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cpss import Hybrid_LISTA_cpss
        model = Hybrid_LISTA_cpss (kwargs['A'], T=config.T, lam=config.lam,
                            percent=config.percent, max_percent=config.max_percent,
                            untied=config.untied , coord=config.coord,
                            scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                            kernel_size=config.kernel_size, feature_map=config.feature_map, 
                            alpha_initial=config.alpha_init)
        
    if config.net == 'Hybrid_LISTA_cpss_FixNetFunc' :
        """Hybrid_LISTA-CPSS_FixNetFunc"""
        config.model = ("Hybrid_LISTA_cpss_FixNetFunc{NetFunc}_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (NetFunc=config.net_func, T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, untiedf=untiedf,
                                 coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        from models.Hybrid_LISTA_cpss_FixNetFunc import Hybrid_LISTA_cpss_FixNetFunc
        model = Hybrid_LISTA_cpss_FixNetFunc (kwargs['A'], T=config.T, lam=config.lam,
                            percent=config.percent, max_percent=config.max_percent,
                            untied=config.untied , coord=config.coord,
                            scope=config.scope, mode=config.w_mode, 
                            alpha_initial=config.alpha_init, Net_Func=config.net_func)

    if config.net == 'TiLISTA':
        """TiLISTA"""
        config.model = ("TiLISTA_T{T}_lam{lam}_p{p}_mp{mp}_"
                        "{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam,
                                 p=config.percent, mp=config.max_percent,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.TiLISTA import TiLISTA
        # Note that TiLISTA is just LISTA-CPSS with tied weight in all layers.
        model = TiLISTA(kwargs['A'], T=config.T, lam=config.lam,
                        percent=config.percent, max_percent=config.max_percent,
                        coord=config.coord, scope=config.scope)

    if config.net == "ALISTA":
        """ALISTA"""
        config.model = ("ALISTA_T{T}_lam{lam}_p{p}_mp{mp}_{W}_{coordf}_{exp_id}"
                        .format(T=config.T, lam=config.lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W),
                                coordf=coordf, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.ALISTA import ALISTA
        model = ALISTA(kwargs['A'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       coord=config.coord, scope=config.scope)
    
    if config.net == "ALISTA_cs_ge":
        """ALISTA_cs_ge"""
        config.model = ("ALISTA_cs_ge_T{T}_lam{lam}_llam{llam}_{coordf}_{exp_id}"
                        .format(T=config.T, lam=config.lam, llam=config.lasso_lam,
                                coordf=coordf, exp_id=config.exp_id))
        from models.ALISTA_cs_ge import ALISTA_cs_ge
        model = ALISTA_cs_ge(kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam,
                       coord=config.coord, scope=config.scope)

    if config.net == "ALISTA_cs":
        """ALISTA_cs"""
        config.model = ("ALISTA_cs_T{T}_lam{lam}_llam{llam}_{W}_{coordf}_{exp_id}"
                        .format(T=config.T, lam=config.lam, llam=config.lasso_lam,
                                W=os.path.basename(config.W),
                                coordf=coordf, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.ALISTA_cs import ALISTA_cs
        model = ALISTA_cs(kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam, W=W,
                       coord=config.coord, scope=config.scope)

    if config.net == "Hybrid_ALISTA":
        """Hybrid_ALISTA"""
        config.model = ("Hybrid_ALISTA_T{T}_lam{lam}_p{p}_mp{mp}_{W}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format(T=config.T, lam=config.lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W), untiedf=untiedf,
                                coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.Hybrid_ALISTA import Hybrid_ALISTA
        model = Hybrid_ALISTA(kwargs['A'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       untied=config.untied, mode=config.w_mode,
                       coord=config.coord, scope=config.scope,
                       conv_num=config.conv_num, kernel_size=config.kernel_size, 
                       feature_map=config.feature_map, alpha_initial=config.alpha_init)
        
    if config.net == "Hybrid_ALISTA_FixNetFunc":
        """Hybrid_ALISTA_FixNetFunc"""
        config.model = ("Hybrid_ALISTA_FixNetFunc{NetFunc}_T{T}_lam{lam}_p{p}_mp{mp}_{W}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format(NetFunc=config.net_func, T=config.T, lam=config.lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W), untiedf=untiedf,
                                coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.Hybrid_ALISTA_FixNetFunc import Hybrid_ALISTA_FixNetFunc
        model = Hybrid_ALISTA_FixNetFunc(kwargs['A'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       untied=config.untied, mode=config.w_mode,
                       coord=config.coord, scope=config.scope,
                       alpha_initial=config.alpha_init, Net_Func=config.net_func)
    
    if config.net == "Hybrid_ALISTA_cs":
        """Hybrid_ALISTA_cs"""
        config.model = ("Hybrid_ALISTA_cs_T{T}_lam{lam}_llam{llam}_p{p}_mp{mp}_{W}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format(T=config.T, lam=config.lam, llam=config.lasso_lam,
                                p=config.percent, mp=config.max_percent,
                                W=os.path.basename(config.W), untiedf=untiedf,
                                coordf=coordf, w_mode=config.w_mode, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.Hybrid_ALISTA_cs import Hybrid_ALISTA_cs
        model = Hybrid_ALISTA_cs(kwargs['Phi'], kwargs['D'], T=config.T, lam=config.lam, W=W,
                       percent=config.percent, max_percent=config.max_percent,
                       untied=config.untied, mode=config.w_mode,
                       coord=config.coord, scope=config.scope,
                       conv_num=config.conv_num, kernel_size=config.kernel_size, 
                       feature_map=config.feature_map, alpha_initial=config.alpha_init)

    if config.net == 'LISTA_cs':
        """LISTA-CS"""
        config.model = ("LISTA_cs_T{T}_lam{lam}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cs import LISTA_cs
        model = LISTA_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                          lam=config.lam, untied=config.untied,
                          coord=config.coord, scope=config.scope)

    if config.net == 'LISTA_ss_cs' :
        """LISTA-SS-CS"""
        config.model = ("LISTA_ss_cs_T{T}_lam{lam}_p{p}_mp{mp}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_ss_cs import LISTA_ss_cs
        model = LISTA_ss_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                             lam=config.lam, percent=config.percent,
                             max_percent=config.max_percent,
                             untied=config.untied, coord=config.coord,
                             scope=config.scope)

    if config.net == 'LISTA_cpss_cs' :
        """LISTA-CPSS-CS"""
        config.model = ("LISTA_cpss_cs_T{T}_lam{lam}_p{p}_mp{mp}_llam{llam}_"
                        "{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf,
                                 exp_id=config.exp_id))
        from models.LISTA_cpss_cs import LISTA_cpss_cs
        model = LISTA_cpss_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                               lam=config.lam, percent=config.percent,
                               max_percent=config.max_percent,
                               untied=config.untied, coord=config.coord,
                               scope=config.scope)

    if config.net == 'Hybrid_LISTA_cpss_cs' :
        """Hybrid_LISTA-CPSS-CS"""
        config.model = ("Hybrid_LISTA_cpss_cs_T{T}_lam{lam}_p{p}_mp{mp}_llam{llam}_"
                        "{untiedf}_{coordf}_W{w_mode}_{exp_id}"
                        .format (T=config.T, lam=config.lam, p=config.percent,
                                 mp=config.max_percent, llam=config.lasso_lam,
                                 untiedf=untiedf, coordf=coordf, w_mode=config.w_mode,
                                 exp_id=config.exp_id))
        from models.Hybrid_LISTA_cpss_cs import Hybrid_LISTA_cpss_cs
        model = Hybrid_LISTA_cpss_cs (kwargs['Phi'], kwargs['D'], T=config.T,
                               lam=config.lam, percent=config.percent,
                               max_percent=config.max_percent,
                               untied=config.untied, coord=config.coord,
                               scope=config.scope, mode=config.w_mode, conv_num=config.conv_num, 
                               kernel_size=config.kernel_size, feature_map=config.feature_map, 
                               alpha_initial=config.alpha_init)

    if config.net == 'LISTA_cp_conv':
        """LISTA-CP-CONV"""
        config.model = ("LISTA_cp_conv_T{T}_lam{lam}_alpha{alpha}_"
                        "sigma{sigma}_{untiedf}_{exp_id}.npz"
                        .format(T=config.T, lam=config.lam, alpha=config.conv_alpha,
                                sigma=config.sigma, untiedf=untiedf, coordf=coordf,
                                exp_id=config.exp_id))
        from models.LISTA_cp_conv import LISTA_cp_conv
        model = LISTA_cp_conv(kwargs['filters'], T=config.T,
                              lam=config.lam, alpha=config.conv_alpha,
                              untied=config.untied, scope=config.scope)

    if config.net == 'ALISTA_conv':
        """ALISTA-CONV"""
        config.model = ("ALISTA_conv_T{T}_lam{lam}_alpha{alpha}_"
                        "sigma{sigma}_{exp_id}.npz"
                        .format(T=config.T, lam=config.lam, alpha=config.conv_alpha,
                                sigma=config.sigma, exp_id=config.exp_id))
        W = np.load(config.W)
        print("Pre-calculated weight W loaded from {}".format(config.W))
        from models.ALISTA_conv import ALISTA_conv
        model = ALISTA_conv(kwargs['filters'], W=W, T=config.T,
                            lam=config.lam, alpha=config.conv_alpha,
                            scope=config.scope)

    if config.net == "AtoW_grad":
        """AtoW_grad"""
        config.model = ("AtoW_grad_eT{eT}_Binit-{Binit}_eta{eta}_loss-{loss}_ps{ps}_lr{lr}_{id}"
                        .format(eT=config.eT, Binit=config.encoder_Binit, eta=config.eta,
                                loss=config.encoder_loss, ps=config.encoder_psigma,
                                lr=config.encoder_pre_lr, id=config.exp_id))
        from models.AtoW_grad import AtoW_grad
        model = AtoW_grad(config.M, config.N, config.eT, Binit=kwargs["Binit"],
                          eta=config.eta, loss=config.encoder_loss,
                          Q=kwargs["Q"], scope=config.scope)

    if config.net == "robust_ALISTA":
        """Robust ALISTA"""
        config.encoder = ("AtoW_grad_eT{eT}_Binit-{Binit}_eta{eta}_loss-{loss}_ps{ps}_lr{lr}_{id}"
                          .format(eT=config.eT, Binit=config.encoder_Binit, eta=config.eta,
                                  loss=config.encoder_loss, ps=config.encoder_psigma,
                                  lr=config.encoder_pre_lr, id=config.exp_id))
        config.decoder = ("ALISTA_robust_T{T}_lam{lam}_p{p}_mp{mp}_{W}_{coordf}_{exp_id}"
                          .format(T=config.T, lam=config.lam,
                                  p=config.percent, mp=config.max_percent,
                                  W=os.path.basename(config.W),
                                  coordf=coordf, exp_id=config.exp_id))

        # set up encoder
        from models.AtoW_grad import AtoW_grad
        encoder = AtoW_grad(config.M, config.N, config.eT, Binit=kwargs["Binit"],
                            eta=config.eta, loss=config.encoder_loss,
                            Q=kwargs["Q"], scope=config.encoder_scope)
        # set up decoder
        from models.ALISTA_robust import ALISTA_robust
        decoder = ALISTA_robust(M=config.M, N=config.N, T=config.T,
                                percent=config.percent, max_percent=config.max_percent,
                                coord=config.coord, scope=config.decoder_scope)

        model_desc = ("robust_" + config.encoder + '_' + config.decoder +
                     "_elr{}_dlr{}_psmax{}_psteps{}_{}"
                     .format(config.encoder_lr, config.decoder_lr,
                             config.psigma_max, config.psteps, config.exp_id))
        model_dir = os.path.join(config.expbase, model_desc)
        config.resfn = os.path.join(config.resbase, model_desc)
        if not os.path.exists(model_dir):
            if config.test:
                raise ValueError("Testing folder {} not existed".format(model_dir))
            else:
                os.makedirs(model_dir)

        config.enc_load = os.path.join(config.expbase, config.encoder)
        config.dec_load = os.path.join(config.expbase, config.decoder.replace("_robust", ""))
        config.encoderfn = os.path.join(model_dir, config.encoder)
        config.decoderfn = os.path.join(model_dir, config.decoder)
        return encoder, decoder


    config.modelfn = os.path.join(config.expbase, config.model)
    config.resfn = os.path.join(config.resbase, config.model)
    print ("model disc:", config.model)

    return model


############################################################
######################   Training    #######################
############################################################

def run_train(config) :
    if config.task_type == "sc":
        run_sc_train(config)
    elif config.task_type == "cs":
        run_cs_train(config)
    elif config.task_type == "denoise":
        run_denoise_train(config)
    elif config.task_type == "encoder":
        run_encoder_train(config)
    elif config.task_type == "robust":
        run_robust_train(config)


def run_sc_train(config) :
    """Load problem."""
    if not os.path.exists(config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Set up model."""
    model = setup_model (config, A=p.A)

    """Set up input."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    y_, x_, y_val_, x_val_ = (
        train.setup_input_sc (
            config.test, p, config.tbs, config.vbs, config.fixval,
            config.supp_prob, config.SNR, config.magdist, **config.distargs))

    """Set up training."""
    # if 'Hybrid_ELISTA' in config.model:
    #     stages = train.setup_sc_training_HELISTA (
    #         model, y_, x_, y_val_, x_val_, None,
    #         config.init_lr, config.decay_rate, config.lr_decay)
    # else:
    stages = train.setup_sc_training (
            model, y_, x_, y_val_, x_val_, None,
            config.init_lr, config.decay_rate, config.lr_decay)


    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training(sess, stages, config.modelfn, config.scope,
                          config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))

    # end of run_sc_train


def run_cs_train (config) :
    """Load dictionary and sensing matrix."""
    # Phi = np.load (config.sensing)['A']
    Phi = np.load (config.sensing)
    D   = np.load (config.dict)

    """Set up model."""
    model = setup_model (config, Phi=Phi, D=D)

    """Set up inputs."""
    y_, f_, y_val_, f_val_ = train.setup_input_cs(config.train_file,
                                                  config.val_file,
                                                  config.tbs, config.vbs)

    """Set up training."""
    if 'cs_ISTANet_Plus' in config.model:
        stages = train.setup_cs_training_ISTANet_Plus (
            model, y_, f_, y_val_, f_val_, None, config.init_lr, config.decay_rate,
            config.lr_decay, config.lasso_lam)
    elif 'ISTA_Net_Plus' in config.model:
        stages = train.setup_cs_training_pure_ISTANet_Plus (
            model, y_, f_, y_val_, f_val_, None, config.init_lr, config.decay_rate,
            config.lr_decay, 0.0) 
    else:
        stages = train.setup_cs_training (
            model, y_, f_, y_val_, f_val_, None, config.init_lr, config.decay_rate,
            config.lr_decay, config.lasso_lam)


    """Start training."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training (sess, stages, config.modelfn, config.scope,
                           config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))

    # end of run_cs_train


def run_denoise_train (config) :
    """Load problem."""
    import utils.prob_conv as problem

    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Set up model."""
    model = setup_model (config, filters=p._fs)

    """Set up input."""
    # training
    clean_ = data.bsd500_denoise_inputs(config.data_folder, config.train_file, config.tbs,
                                        config.height_crop, config.width_crop, config.num_epochs)
    clean_.set_shape((config.tbs, *clean_.get_shape()[1:],))
    # validation
    clean_val_ = data.bsd500_denoise_inputs(config.data_folder, config.val_file, config.vbs,
                                            config.height_crop, config.width_crop, 1)
    clean_val_.set_shape((config.vbs, *clean_val_.get_shape()[1:],))
    # add noise
    noise_ = tf.random_normal(clean_.shape, stddev=config.denoise_std,
                              dtype=tf.float32)
    noise_val_ = tf.random_normal(clean_val_.shape, stddev=config.denoise_std,
                                  dtype=tf.float32)
    noisy_ = clean_ + noise_
    noisy_val_= clean_val_ + noise_val_

    # fix validation set
    with tf.name_scope ('input'):
        clean_val_ = tf.get_variable(name='clean_val',
                                     dtype=tf.float32,
                                     initializer=clean_val_)
        noisy_val_ = tf.get_variable(name='noisy_val',
                                     dtype=tf.float32,
                                     initializer=noisy_val_)
    """Set up training."""
    stages = train.setup_denoise_training(
        model, noisy_, clean_, noisy_val_, clean_val_,
        None, config.init_lr, config.decay_rate, config.lr_decay)

    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training(sess, stages, config.modelfn, config.scope,
                          config.val_step, config.maxit, config.better_wait)

        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))

    # end of run_denoise_train


def run_encoder_train(config):
    """Load problem."""
    if not os.path.exists(config.probfn):
        raise ValueError("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Load the Q reweighting matrix."""
    if config.Q is None: # use default Q reweighting matrix
        if "re" in config.encoder_loss: # if using reweighted loss
            Q = np.sqrt((np.ones(shape=(config.N, config.N), dtype=np.float32) +
                         np.eye(config.N, dtype=np.float32) * (config.N - 2)))
        else:
            Q = None
    elif os.path.exists(config.Q) and config.Q.endswith(".npy"):
        Q = np.load(config.Q)
        assert Q.shape == (config.N, config.N)
    else:
        raise ValueError("Invalid parameter `--Q`\n"
            "A valid `--Q` parameter should be one of the following:\n"
            "    1) omitted for default value as in the paper;\n"
            "    2) path/to/your/npy/file that contains your Q matrix.\n")

    """Binit matrix."""
    if config.encoder_Binit == "default":
        Binit = p.A
    elif config.Binit in ["uniform", "normal"]:
        pass
    else:
        raise ValueError("Invalid parameter `--Binit`\n"
            "A valid `--Binit` parameter should be one of the following:\n"
            "    1) omitted for default value `p.A`;\n"
            "    2) `normal` or `uniform`.\n")

    """Set up model."""
    model = setup_model(config, Binit=Binit, Q=Q)
    print("The trained model will be saved in {}".format(config.model))

    """Set up training."""
    from utils.tf import get_loss_func, bmxbm, mxbm
    with tf.name_scope ('input'):

        A_ = tf.constant(p.A, dtype=tf.float32)
        perturb_ = tf.random.normal(shape=(config.Abs, config.M, config.N),
                                    mean=0.0, stddev=config.encoder_psigma,
                                    dtype=tf.float32)
        Ap_ = A_ + perturb_
        Ap_ = Ap_ / tf.sqrt(tf.reduce_sum(tf.square( Ap_ ), axis=1, keepdims=True))
        Apt_ = tf.transpose(Ap_, [0,2,1])
        W_ = model.inference(Ap_)

        """Set up loss."""
        eye_ = tf.eye(config.N, batch_shape=[config.Abs], dtype=tf.float32)
        residual_ = bmxbm(Apt_, W_, batch_first=True) - eye_
        loss_func = get_loss_func(config.encoder_loss, model._Q_)
        loss_ = loss_func(residual_)

        # fix validation set
        Ap_val_ = tf.get_variable(name='Ap_val', dtype=tf.float32,
                                  initializer=Ap_, trainable=False)
        Apt_val_ = tf.transpose(Ap_val_, [0,2,1])
        W_val_ = model.inference(Ap_val_)
        # validation loss
        residual_val_ = bmxbm(Apt_val_, W_val_, batch_first=True) - eye_
        loss_val_ = loss_func(residual_val_)

    """Set up optimizer."""
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(config.encoder_lr, global_step,
                                    5000, 0.75, staircase=True)
    learning_step = (tf.train.AdamOptimizer(lr)
                     .minimize(loss_, global_step=global_step))

    # create session and initialize the graph
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()
        for i in range (config.maxit):
            # training step
            _, loss = sess.run([learning_step, loss_])

            # validation step
            if i % config.val_step == 0:
                # validation step
                loss_val = sess.run(loss_val_)
                sys.stdout.write (
                    "\ri={i:<7d} | train_loss={train_loss:.6f} | "
                    "loss_val={loss_val:.6f}"
                    .format(i=i, train_loss=loss, loss_val=loss_val))
                sys.stdout.flush()

        # end timer
        end = time.time()
        elapsed = end - start
        print("elapsed time of training = " + str(timedelta(seconds=elapsed)))

        train.save_trainable_variables (sess, config.modelfn, config.scope)
        print("model saved to {}".format(config.modelfn))

    # end of run_encoder_train


def run_robust_train(config):
    """Load problem."""
    if not os.path.exists(config.probfn):
        raise ValueError("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Set up input."""
    # `psigma` is a list of standard deviations for curriculum learning
    psigmas = np.linspace(0, config.psigma_max, config.psteps)[1:]
    psigma_ = tf.placeholder(dtype=tf.float32, shape=())
    with tf.name_scope ('input'):
        Ap_, y_, x_ = train.setup_input_robust(p.A, psigma_, config.msigma,
                                               p.pnz, config.Abs, config.xbs)
        if config.net != "robust_ALISTA":
            # If not joint robust training
            # reshape y_ into shape (m, Abs * xbs)
            # reshape x_ into shape (n, Abs * xbs)
            y_ = tf.reshape(tf.transpose(y_, [1, 0, 2]), (config.M, -1))
            x_ = tf.reshape(tf.transpose(x_, [1, 0, 2]), (config.N, -1))
        # fix validation set
        Ap_val_ = tf.get_variable(name="Ap_val", dtype=tf.float32, initializer=Ap_)
        y_val_ = tf.get_variable(name="y_val", dtype=tf.float32, initializer=y_)
        x_val_ = tf.get_variable(name="x_val", dtype=tf.float32, initializer=x_)

    """Set up model."""
    if config.net == "robust_ALISTA":
        """Load the Q reweighting matrix."""
        if config.Q is None: # use default Q reweighting matrix
            if "re" in config.encoder_loss: # if using reweighted loss
                Q = np.sqrt((np.ones(shape=(config.N, config.N), dtype=np.float32) +
                             np.eye(config.N, dtype=np.float32) * (config.N - 2)))
            else:
                Q = None
        elif os.path.exists(config.Q) and config.Q.endswith(".npy"):
            Q = np.load(config.Q)
            assert Q.shape == (config.N, config.N)
        else:
            raise ValueError("Invalid parameter `--Q`\n"
                "A valid `--Q` parameter should be one of the following:\n"
                "    1) omitted for default value as in the paper;\n"
                "    2) path/to/your/npy/file that contains your Q matrix.\n")

        """Binit matrix."""
        if config.encoder_Binit == "default":
            Binit = p.A
        elif config.Binit in ["uniform", "normal"]:
            pass
        else:
            raise ValueError("Invalid parameter `--Binit`\n"
                "A valid `--Binit` parameter should be one of the following:\n"
                "    1) omitted for default value `p.A`;\n"
                "    2) `normal` or `uniform`.\n")

        encoder, decoder = setup_model(config, Q=Q, Binit=Binit)
        W_ = encoder.inference(Ap_)
        W_val_ = encoder.inference(Ap_val_)
        xh_ = decoder.inference(y_, Ap_, W_, x0_=None)[-1]
        xh_val_ = decoder.inference(y_val_, Ap_val_, W_val_, x0_=None)[-1]
    else:
        decoder = setup_model(config, A=p.A)
        xh_ = decoder.inference(y_, None)[-1]
        xh_val_ = decoder.inference(y_val_, None)[-1]
        config.dec_load = config.modelfn
        config.decoder = (
            "robust_" + config.model + '_ps{ps}_nsteps{nsteps}_ms{ms}_lr{lr}'
            .format(ps=config.psigma_max, nsteps=config.psteps,
                    ms=config.msigma, lr=config.decoder_lr))
        config.decoderfn = os.path.join(config.expbase, config.decoder)
        print("\npretrained decoder loaded from {}".format(config.modelfn))
        print("trained augmented model will be saved to {}".format(config.decoderfn))

    """Set up loss."""
    loss_ = tf.nn.l2_loss (xh_ - x_)
    nmse_denom_ = tf.nn.l2_loss (x_)
    nmse_ = loss_ / nmse_denom_
    db_ = 10.0 * tf.log (nmse_) / tf.log (10.0)
    # validation
    loss_val_ = tf.nn.l2_loss (xh_val_ - x_val_)
    nmse_denom_val_ = tf.nn.l2_loss (x_val_)
    nmse_val_ = loss_val_ / nmse_denom_val_
    db_val_ = 10.0 * tf.log (nmse_val_) / tf.log (10.0)

    """Set up optimizer."""
    global_step_ = tf.Variable (0, trainable=False)
    if config.net == "robust_ALISTA":
        """Encoder and decoder apply different initial learning rate."""
        # get trainable variable for de encoder and decoder
        encoder_variables_ = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=config.encoder_scope)
        decoder_variables_ = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=config.decoder_scope)
        trainable_variables_ = encoder_variables_ + decoder_variables_
        # calculate gradients w.r.t. all trainable variables in the model
        grads_ = tf.gradients (loss_, trainable_variables_)
        encoder_grads_ = grads_[:len (encoder_variables_)]
        decoder_grads_ = grads_[len (encoder_variables_):]
        # define learning rates for optimizers over two parts
        global_step_ = tf.Variable (0, trainable=False)
        encoder_lr_ = tf.train.exponential_decay(
                config.encoder_lr, global_step_, 5000, 0.75, staircase=False)
        encoder_opt_ = tf.train.AdamOptimizer(encoder_lr_)
        decoder_lr_ = tf.train.exponential_decay(
                config.decoder_lr, global_step_, 5000, 0.75, staircase=False)
        decoder_opt_ = tf.train.AdamOptimizer(decoder_lr_)
        # define training operator
        encoder_op_ = encoder_opt_.apply_gradients(
                zip(encoder_grads_, encoder_variables_))
        decoder_op_ = decoder_opt_.apply_gradients(
                zip(decoder_grads_, decoder_variables_))
        learning_step_ = tf.group(encoder_op_, decoder_op_)
    else:
        lr_ = tf.train.exponential_decay(config.decoder_lr, global_step_,
                                         5000, 0.75, staircase=False)
        learning_step_ = (tf.train.AdamOptimizer(lr_)
                          .minimize(loss_, global_step=global_step_))

    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer (),
                  feed_dict={psigma_: psigmas[0]})

        # load pre-trained model(s)
        if config.net == "robust_ALISTA":
            encoder.load_trainable_variables(sess, config.enc_load)
        decoder.load_trainable_variables(sess, config.dec_load)

        # start timer
        start = time.time ()

        for psigma in psigmas:
            print ('\ncurrent sigma: {}'.format (psigma))
            global_step_.initializer.run ()

            for i in range (config.maxit):
                db, loss, _ = sess.run([db_, loss_, learning_step_],
                                       feed_dict={psigma_: psigma})

                if i % config.val_step == 0:
                    db_val, loss_val = sess.run([db_val_, loss_val_],
                                                feed_dict={psigma_: psigma})
                    sys.stdout.write(
                        "\ri={i:<7d} | loss_train={loss_train:.6f} | "
                        "db_train={db_train:.6f} | loss_val={loss_val:.6f} | "
                        "db_val={db_val:.6f}".format(
                            i=i, loss_train=loss, db_train=db,
                            loss_val=loss_val, db_val=db_val))
                    sys.stdout.flush()

        if config.net == "robust_ALISTA":
            encoder.save_trainable_variables(sess, config.encoderfn)
        decoder.save_trainable_variables(sess, config.decoderfn)

        # end timer
        end = time.time()
        elapsed = end - start
        print("elapsed time of training = " + str(timedelta(seconds=elapsed)))

    # end of run_robust_train


############################################################
######################   Testing    ########################
############################################################

def run_test (config):
    if config.task_type == "sc":
        run_sc_test (config)
    elif config.task_type == "sc_time":
        run_sc_time_test (config)
    elif config.task_type == 'sc_conv':
        run_sc_conv_test (config)
    elif config.task_type == "cs":
        run_cs_test (config)
    elif config.task_type == "denoise":
        run_denoise_test(config)
    elif config.task_type == "robust":
        run_robust_test(config)

def run_sc_test (config) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    zeros_pos = np.logical_not(xt)
    not_zeros_pos = np.logical_not(zeros_pos)
    # xt_2norm = np.linalg.norm(xt, ord=2)

    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))


    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)

    if config.model[0] == 'H':
        print(config.model)
        if 'ISTA_ada_fixT' in config.model:
            theta_list = model.thetas_     
            delta_list = model.delta_total_
            alpha_list = model.alpha_list
            eta_list = model.eta_list
            lam_f_list = model.lam_f_list

        elif 'ISTA_ada_freeT' in config.model:
            theta_list = model.thetas_     
            eta_list = model.eta_list
            lam_f_list = model.lams_

        elif 'LISTA' in config.model and 'ELISTA' not in config.model:
            theta1_list = model.thetas1_ 
            theta2_list = model.thetas2_
            alpha_list = model.alphas_raw
            u_list = model.uhs_
            
        elif 'ELISTA' in config.model:
            alpha_list = model._alpha
            u_list = model.uhs_
        

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        nmse_denom = np.sum (np.square (xt))
        supp_gt = xt != 0

        lnmse  = []
        lspar  = []
        lsperr = []
        lflspo = []
        lflsne = []

        false_positive = []
        true_positive = []

        # test model
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})
            xh_2norm = np.linalg.norm(xh, ord=2)

            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print (nmse_dB)

            lnmse.append (nmse_dB)

            supp = xh != 0.0
            # intermediate sparsity
            spar = np.sum (supp , axis=0)
            lspar.append (spar)

            # support error
            sperr = np.logical_xor(supp, supp_gt)
            lsperr.append (np.sum (sperr , axis=0))

            # false positive
            flspo = np.logical_and (supp , np.logical_not (supp_gt))
            lflspo.append (np.sum (flspo , axis=0))

            # false negative
            flsne = np.logical_and (supp_gt , np.logical_not (supp))
            lflsne.append (np.sum (flsne , axis=0))
            
            if xh_2norm == 0:
                false_positive.append(0)
                true_positive.append(0)
            else:
                false_positive.append(np.square(np.linalg.norm(zeros_pos * xh, ord=2) / xh_2norm ))
                true_positive.append(np.square(np.linalg.norm(not_zeros_pos * xh, ord=2) / xh_2norm ))

        if config.model[0] == 'H':

            if 'ISTA_ada_fixT' in config.model:
                lam_f_, eta_, del_, t_, al,  = sess.run ([lam_f_list, eta_list, delta_list, theta_list, alpha_list] , feed_dict={label_:xt})
                Lam_ = [abs(l) for l in lam_f_]
                Theta_ = [abs(t) for t in t_]

                print('Lam_Theta:', np.multiply(np.array(Lam_),np.array(Theta_)).tolist())
                print('delta:', [0.25/(1+np.exp(-1*d))+0.25 for d in del_])
                print('Alpha:', al)
                print('eta:', eta_)
            
            elif 'ISTA_ada_freeT' in config.model:
                lam_f_, eta_, t_, = sess.run ([lam_f_list, eta_list, theta_list] , feed_dict={label_:xt})
                
                Lam_ = [abs(l) for l in lam_f_]
                Theta_ = [abs(t) for t in t_]

                print('Lam_Theta:', np.multiply(np.array(Lam_),np.array(Theta_)).tolist()) 
                print('eta:', eta_)

            elif 'LISTA' in config.model and 'ELISTA' not in config.model:
                t1, t2, al, u = sess.run ([theta1_list, theta2_list, alpha_list, u_list] , feed_dict={label_:xt})
                alpha = []
                u_1norm = []
                
                for i in range(config.T):
                    alpha.append(1 - 1.0/(1+np.exp(-al[i]))*abs(t1[i])/(abs(t1[i])+abs(t2[i])))
                    u_1norm.append(np.linalg.norm(u[i], ord=1)/1000)
                    
                print('Theta1:', [abs(t) for t in t1])
                print('Theta2:', [abs(t) for t in t2])
                print('Alpha:', alpha)
                print('u_1norm', u_1norm)
                print('x_1norm', np.linalg.norm(xt, ord=1)/1000)
                
            elif 'ELISTA' in config.model:
                al, u = sess.run ([alpha_list, u_list] , feed_dict={label_:xt})
                alpha = []
                u_1norm = []
                
                for i in range(config.T):
                    alpha.append(al[i])
                    u_1norm.append(np.linalg.norm(u[i], ord=1)/1000)

                print('Alpha:', alpha)
                print('u_1norm', u_1norm)
                print('x_1norm', np.linalg.norm(xt, ord=1)/1000)
                
        print('false_positive:')
        for item in false_positive:
            print(item)
        print('true_positive:')
        for item in true_positive:
            print(item)

    res = dict (nmse=np.asarray  (lnmse),
                spar=np.asarray  (lspar),
                sperr=np.asarray (lsperr),
                flspo=np.asarray (lflspo),
                flsne=np.asarray (lflsne))

    np.savez (config.resfn , **res)
    
    # end of test


def run_sc_time_test (config) :
    """
    Test model and record time.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    zeros_pos = np.logical_not(xt)
    not_zeros_pos = np.logical_not(zeros_pos)
    # xt_2norm = np.linalg.norm(xt, ord=2)

    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))


    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)
        

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        print(config.net)
        # load model
        if config.net not in  ['ISTA', 'FISTA', 'ADMM']:
            model.load_trainable_variables (sess , config.modelfn)

        # test model
        elapsed_time = []
        for i in range(100):
            start = time.time()
            _ = sess.run (xhs_[-1], feed_dict={label_:xt})
            end = time.time()
            print(end - start)
            if i != 0:
                elapsed_time.append(end - start)
            
        elapsed = np.mean(elapsed_time)
        print("elapsed time of testing = " + str(elapsed))
        ave_elapsed = elapsed / (config.T)
        print("average elapsed time of testing = " + str(ave_elapsed))
        

def run_sc_conv_test (config) :
    """
    Test conventional model and record time.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    zeros_pos = np.logical_not(xt)
    not_zeros_pos = np.logical_not(zeros_pos)
    # xt_2norm = np.linalg.norm(xt, ord=2)

    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (
        train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist, **config.distargs))


    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)
        

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        dir = './conventional_algorithm_results/'+str(config.net)+'_'+str(config.T)+'_'+str(config.lam)+'.txt'
        nmse_denom = np.sum (np.square (xt))
        
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})
            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print (nmse_dB)
            write_txt (dir, nmse_dB)
        

def run_cs_test (config) :
    from utils.cs import imread_CS_py, img2col_py, col2im_CS_py
    from skimage.io import imsave
    """Load dictionary and sensing matrix."""
    # Phi = np.load (config.sensing) ['A']
    Phi = np.load (config.sensing)
    D   = np.load (config.dict)
    
    print(Phi.shape)
    print(D.shape)
    
    # loading compressive sensing settings
    M = Phi.shape [0]
    F = Phi.shape [1]
    N = D.shape [1]
    # assert M == config.M and F == config.F and N == config.N
    patch_size = int (np.sqrt (F))
    assert patch_size ** 2 == F


    """Set up model."""
    model = setup_model (config, Phi=Phi, D=D)

    """Inference."""
    y_ = tf.placeholder (shape=(M, None), dtype=tf.float32)
    
    if 'cs_ISTANet_Plus' in config.model:
        _, fhs_, _ = model.inference (y_, None)
    elif 'ISTA_Net_Plus' in config.model:
        fhs_, _ = model.inference (y_, None)
    else:
        _, fhs_ = model.inference (y_, None)


    """Start testing."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:

        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        # calculate average NMSE and PSRN on test images
        test_dir = './data/test_images/'
        test_files = os.listdir (test_dir)
        avg_nmse = 0.0
        avg_psnr = 0.0
        avg_ssim = 0.0
        overlap = 0
        stride = patch_size - overlap
        out_dir = "./data/recon_images/" + config.model
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if 'joint' in config.net :
            D = sess.run (model.D_)
            
        for test_fn in test_files :
            # read in image
            prefix = test_fn[:-4]
            test_fn = os.path.join (test_dir, test_fn)
            test_im, H, W, test_im_pad, H_pad, W_pad = \
                    imread_CS_py (test_fn, patch_size, stride)
            test_fs = img2col_py (test_im_pad, patch_size, stride)

            # remove dc from features
            test_dc = np.mean (test_fs, axis=0, keepdims=True)
            test_cfs = test_fs - test_dc
            test_cfs = np.asarray (test_cfs) / 255.0

            # sensing signals
            test_ys = np.matmul (Phi, test_cfs)
            num_patch = test_ys.shape [1]

            rec_cfs = sess.run (fhs_ [-1], feed_dict={y_: test_ys})
            rec_fs  = rec_cfs * 255.0 + test_dc

            # patch-level NMSE
            patch_err = np.sum (np.square (rec_fs - test_fs))
            patch_denom = np.sum (np.square (test_fs))
            avg_nmse += 10.0 * np.log10 (patch_err / patch_denom)

            rec_im = col2im_CS_py (rec_fs, patch_size, stride,
                                   H, W, H_pad, W_pad)

            # image-level PSNR
            image_mse = np.mean (np.square (np.clip(rec_im, 0.0, 255.0) - test_im))
            current_psnr = 10.0 * np.log10 (255.**2 / image_mse)
            avg_psnr += current_psnr

            # Calculate SSIM
            rec_total = tf.expand_dims(tf.constant(np.clip(rec_im, 0.0, 255.0), dtype=tf.float32), -1)
            test_total = tf.expand_dims(tf.constant(test_im, dtype=tf.float32), -1)
            current_ssim = sess.run(tf.image.ssim(rec_total, test_total, 255))
            avg_ssim += current_ssim

            out_fn = prefix + "_recon_{}_P_{:.4f}_S_{:.5f}.png".format(config.sample_rate, current_psnr, current_ssim)
            out_fn = os.path.join(out_dir, out_fn)
            imwrite(out_fn, rec_im)

        # if config.model[0] == 'A':
        #     w_ = sess.run (learned_w) 
        #     np.save('W_cs.npy', w_)

    
    num_test_ims = len (test_files)
    print ('Average Patch-level NMSE is {}'.format (avg_nmse / num_test_ims))
    print ('Average Image-level PSNR is {}'.format (avg_psnr / num_test_ims))
    print ('Average Image-level SSIM is {}'.format (avg_ssim / num_test_ims))

    # end of cs_testing


def run_denoise_test(config) :
    import glob
    from PIL import Image
    """Load problem."""
    import utils.prob_conv as problem

    if not os.path.exists(config.probfn):
        raise ValueError("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Set up model."""
    model = setup_model(config, filters=p._fs)

    """Set up input."""
    orig_clean_ = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 1))
    clean_ = orig_clean_ * (1.0 / 255.0)
    mean_ = tf.reduce_mean(clean_, axis=(1,2,3,), keepdims=True)
    demean_ = clean_ - mean_

    """Add noise."""
    noise_ = tf.random_normal (tf.shape (demean_), stddev=config.denoise_std,
                               dtype=tf.float32)
    noisy_ = demean_ + noise_

    """Inference."""
    _, recons_ = model.inference(noisy_, None)
    recon_ = recons_[-1]
    # denormalize
    recon_ = (recon_ + mean_) * 255.0

    """PSNR."""
    mse2_ = tf.reduce_mean(tf.square(orig_clean_ - recon_), axis=(1,2,3,))
    psnr_ = 10.0 * tf.log(255.0 ** 2 / mse2_) / tf.log (10.0)
    avg_psnr_ = tf.reduce_mean(psnr_)

    """Load test images."""
    test_images = []
    filenames = []
    types = ("*.tif", "*.png", "*.jpg", "*.gif",)
    for type in types:
        filenames.extend(glob.glob(os.path.join(config.test_dir, type)))
    for filename in filenames:
        im = Image.open(filename)
        if im.size != (256, 256):
            im = im.resize((256, 256))
        test_images.append(np.asarray (im).astype(np.float32))
    test_images = np.asarray(test_images).reshape((-1, 256, 256, 1))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        # graph initialization
        sess.run(tf.global_variables_initializer())

        # load model
        model.load_trainable_variables(sess, config.modelfn)

        # testing
        psnr, avg_psnr = sess.run([psnr_, avg_psnr_],
                                  feed_dict={orig_clean_:test_images})
        print('file names\t| PSNR/dB')
        for fname, p in zip(filenames, psnr):
            print(os.path.basename (fname), '\t', p)

        print("average PSNR = {} dB".format(avg_psnr))
        print("full PSNR records on testing set are stored in {}".format(config.resfn))
        np.save(config.resfn, psnr)

        sum_time = 0.0
        ntimes = 200
        for i in range(ntimes):
            # start timer
            start = time.time()
            # testing
            sess.run(recon_, feed_dict={orig_clean_:test_images})
            # end timer
            end = time.time()
            sum_time = sum_time + end - start
        print("average elapsed time for one image inference = " +
              str(timedelta(seconds=sum_time/ntimes/test_images.shape[0])))

        # start timer
        start = time.time()

    # end of run_denoise_test


def run_robust_test(config):
    """Load problem."""
    print(config.probfn)
    if not os.path.exists(config.probfn):
        raise ValueError("Problem file not found.")
    else:
        p = problem.load_problem(config.probfn)

    """Set tesing data."""
    test_As = np.load('./data/robust_test_A.npz')
    x = np.load('./data/xtest_n500_p10.npy')

    """Set up input."""
    psigmas = sorted([float(k) for k in test_As.keys()])
    psigma_ = tf.placeholder(dtype=tf.float32, shape=())
    with tf.name_scope ('input'):
        Ap_ = tf.placeholder (dtype=tf.float32, shape=(250, 500))
        x_ = tf.placeholder (dtype=tf.float32, shape=(500, None))
        ## measure y_ from x_ using Ap_
        y_ = tf.matmul (Ap_, x_)

    """Set up model."""
    if config.net == "robust_ALISTA":
        """Load the Q reweighting matrix."""
        if config.Q is None: # use default Q reweighting matrix
            if "re" in config.encoder_loss: # if using reweighted loss
                Q = np.sqrt((np.ones(shape=(config.N, config.N), dtype=np.float32) +
                             np.eye(config.N, dtype=np.float32) * (config.N - 2)))
            else:
                Q = None
        elif os.path.exists(config.Q) and config.Q.endswith(".npy"):
            Q = np.load(config.Q)
            assert Q.shape == (config.N, config.N)
        else:
            raise ValueError("Invalid parameter `--Q`\n"
                "A valid `--Q` parameter should be one of the following:\n"
                "    1) omitted for default value as in the paper;\n"
                "    2) path/to/your/npy/file that contains your Q matrix.\n")

        """Binit matrix."""
        if config.encoder_Binit == "default":
            Binit = p.A
        elif config.Binit in ["uniform", "normal"]:
            pass
        else:
            raise ValueError("Invalid parameter `--Binit`\n"
                "A valid `--Binit` parameter should be one of the following:\n"
                "    1) omitted for default value `p.A`;\n"
                "    2) `normal` or `uniform`.\n")

        encoder, decoder = setup_model(config, Q=Q, Binit=Binit)
        W_ = tf.squeeze(encoder.inference(tf.expand_dims(Ap_, axis=0)), axis=0)
        xh_ = decoder.inference(y_, Ap_, W_, x0_=None)[-1]
    else:
        decoder = setup_model(config, A=p.A)
        xh_ = decoder.inference(y_, None)[-1]
        config.decoder = (
            "robust_" + config.model + '_ps{ps}_nsteps{nsteps}_ms{ms}_lr{lr}'
            .format(ps=config.psigma_max, nsteps=config.psteps,
                    ms=config.msigma, lr=config.decoder_lr))
        config.decoderfn = os.path.join(config.expbase, config.decoder)
        print("\ntrained augmented model loaded from {}".format(config.decoderfn))

    """Set up loss."""
    loss_ = tf.nn.l2_loss (xh_ - x_)
    nmse_denom_ = tf.nn.l2_loss (x_)
    nmse_ = loss_ / nmse_denom_
    db_ = 10.0 * tf.log (nmse_) / tf.log (10.0)

    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer (),
                  feed_dict={psigma_: psigmas[0]})

        # load pre-trained model(s)
        if config.net == "robust_ALISTA":
            encoder.load_trainable_variables(sess, config.encoderfn)
        decoder.load_trainable_variables(sess, config.decoderfn)

        # start timer
        start = time.time ()

        res = dict (sigmas=np.array (psigmas))
        avg_dBs = []
        print ('sigma\tnmse')

        sum_time = 0.0
        tcounter = 0

        for psigma in psigmas:
            Aps = test_As[str(psigma)]
            sum_dB = 0.0
            counter = 0
            for Ap in Aps:
                db = sess.run(db_, feed_dict={x_:x, Ap_:Ap})

                # start timer
                start = time.time ()
                # inference
                sess.run (xh_, feed_dict={x_:x, Ap_:Ap})
                # end timer
                end = time.time ()
                elapsed = end - start
                sum_time = sum_time + elapsed
                tcounter = tcounter + 1

                sum_dB  = sum_dB + db
                counter = counter + 1
            avg_dB = sum_dB / counter
            print(psigma, '\t', avg_dB)
            avg_dBs.append (avg_dB)

        print("average elapsed time of inference =", str (timedelta (seconds=sum_time/tcounter)))

        res['avg_dBs'] = np.asarray(avg_dBs)
        print('saving results to', config.resfn)
        np.savez (config.resfn, **res)

    # end of run_robust_test

def write_txt (dir, data_save):
    data = open(dir, 'a')
    data.write(str(data_save))
    data.write('\n')
    # data.close()

############################################################
#######################    Main    #########################
############################################################

def main ():
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.test:
        run_test (config)
    else:
        run_train (config)
    # end of main

if __name__ == "__main__":
    main ()

