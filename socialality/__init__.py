"""
@Author: Ziqian Zou
@Date: 2026-01-28 17:40:27
@LastEditors: Ziqian Zou
@LastEditTime: 2026-07-24 09:45:27
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""
import qpid

from .__args import SocialalityArgs
from .ev_sa import EVSocialality, EVSocialalityModel
from .model import Socialality, SocialalityModel
from .re_sa import ResonanceSA, ResonanceSAModel
from .team_group_mask import team_group_mask_handler
from .tran_sa import TransformerSA, TransformerSAModel

qpid.register(
    sa=[Socialality, SocialalityModel],
    transa=[TransformerSA, TransformerSAModel],
    resa=[ResonanceSA, ResonanceSAModel],
    evsa=[EVSocialality, EVSocialalityModel],
    )
qpid.register_args(SocialalityArgs, 'sa Args')
qpid.register_input_type('TEAM_GROUP_MASK', team_group_mask_handler)
