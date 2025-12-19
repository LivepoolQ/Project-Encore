"""
@Author: Ziqian Zou
@Date: 2025-12-11 17:21:42
@LastEditors: Ziqian Zou
@LastEditTime: 2025-12-19 17:02:12
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2025 Ziqian Zou, All Rights Reserved.
"""
import numpy as np

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class RangerArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """

        """
        return self._arg('Kc', 20, argtype=STATIC)
    
    @property
    def mem_step(self) -> int:
        """
        
        """
        return self._arg('mem_step', 8, argtype=STATIC)
    
    @property
    def use_group(self) -> int:
        """
        Choose whether to use pedestrian groups when calculating SocialCircle.
        """
        return self._arg('use_group', 0, argtype=STATIC, desc_in_model_summary='use kernel function')
    
    @property
    def group_distance(self) -> int:
        return self._arg('group_distance', 6, argtype=STATIC)
    
    @property
    def output_units(self) -> int:
        """
        Set number of the output units of trajectory encoding.
        """
        return self._arg('output_units', 32, argtype=STATIC)
    
    @property
    def generation_num(self) -> int:
        """
        Number of multi-style generation.
        """
        return self._arg('generation_num', 20, argtype=STATIC)
    
    @property
    def view_angle(self) -> float:
        """
        Value of conception view field.
        """
        return self._arg('view_angle', np.pi, argtype=STATIC)
    
    @property
    def ego_loss_ratio(self) -> float:
        """
        Ratio of ego loss when computing sum of l2 loss and ego loss.
        """
        return self._arg('ego_loss_ratio', 0.4, argtype=STATIC, desc_in_model_summary='ego loss ratio')
    
    @property
    def l2_loss_ratio(self) -> float:
        """
        Ratio of l2 loss when computing sum of l2 loss and ego loss.
        """
        return self._arg('l2_loss_ratio', 1.0, argtype=STATIC, desc_in_model_summary='l2 loss ratio')
    
    @property
    def insights_num(self) -> int:
        """
        Number of insights.
        """
        return self._arg('insights_num', 3, argtype=STATIC, desc_in_model_summary='insights num')
    
    @property
    def use_ego_tran(self) -> int:
        """
        (bool) Choose whether to use transformer in egopredictor.
        """
        return self._arg('use_ego_tran', 0, argtype=STATIC)
    
    @property
    def use_lite_egopredictor(self) -> int:
        """
        (bool) Choose whether to use linear prediction in egopredictor.
        """
        return self._arg('use_lite_egopredictor', 0, argtype=STATIC, other_names=['use_ego_linear'])
    
    @property
    def group_type(self) -> int:
        """
        Choose which group method to use, including `[0, 1, 2]`:
        - `0`: Vanilla ;
        - `1`: TODO
        - `2`: TODO
        """
        return self._arg('group_type', 0, argtype=STATIC, desc_in_model_summary='group type')
