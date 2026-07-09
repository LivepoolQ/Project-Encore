"""
@Author: Ziqian Zou
@Date: 2026-01-28 17:40:27
@LastEditors: Ziqian Zou
@LastEditTime: 2026-07-09 09:37:19
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""
import qpid

from .__args import SocialalityArgs
from .model import Socialality, SocialalityModel
from .team_group_mask import team_group_mask_handler

qpid.register(sa=[Socialality, SocialalityModel])
qpid.register_args(SocialalityArgs, 'sa Args')
qpid.register_input_type('TEAM_GROUP_MASK', team_group_mask_handler)
