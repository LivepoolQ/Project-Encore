"""
@Author: Ziqian Zou
@Date: 2025-12-11 17:21:42
@LastEditors: Ziqian Zou
@LastEditTime: 2026-01-13 20:05:09
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
        return self._arg('ego_loss_ratio', 0.4, argtype=STATIC,
                         desc_in_model_summary=('Ego predictor', 'ego_loss ratio'))

    @property
    def l2_loss_ratio(self) -> float:
        """
        Ratio of l2 loss when computing sum of l2 loss and ego loss.
        """
        return self._arg('l2_loss_ratio', 1.0, argtype=STATIC,
                         desc_in_model_summary=('Ego predictor', 'l2_loss ratio'))

    @property
    def insights_num(self) -> int:
        """
        Number of insights.
        """
        return self._arg('insights_num', 3, argtype=STATIC,
                         desc_in_model_summary=('Ego predictor', 'insights num'))

    @property
    def encode_agent_types(self) -> int:
        """
        Choose whether to encode the type name of each agent.
        It is mainly used in multi-type-agent prediction scenes, providing
        a unique type-coding for each type of agents when encoding their
        trajectories.
        """
        return self._arg('encode_agent_types', 0, argtype=STATIC)

    @property
    def ego_predictor_type(self) -> str:
        """
        Choose which kind of backbones ego predictor will use.
        - `linear`:
        - `fc`:
        - `tran`:
        """
        return self._arg('ego_predictor_type', 'tran', argtype=DYNAMIC,
                         desc_in_model_summary=('Ego predictor', 'type'))

    @property
    def ego_t_h(self) -> int:
        """
        Observation time steps calculated in ego predictor.
        """
        return self._arg('ego_t_h', -1, argtype=STATIC,
                         desc_in_model_summary=('Ego predictor', 'ego_t_h'))

    @property
    def ego_t_f(self) -> int:
        """
        Prediction time steps calculated in ego predictor.
        """
        return self._arg('ego_t_f', -1, argtype=STATIC,
                         desc_in_model_summary=('Ego predictor', 'ego_t_f'))

    @property
    def group_type(self) -> int:
        """
        Choose which group method to use, including `[0, 1, 2]`:
        - `0`: Vanilla ;
        - `1`: TODO Model;
        - `2`: TODO.
        """
        return self._arg('group_type', 1, argtype=STATIC, desc_in_model_summary='group type')

    @property
    def ego_capacity(self) -> int:
        """
        Number of neighbors to be carefully considered in ego predictor.
        """
        return self._arg('ego_capacity', -1, DYNAMIC,
                         desc_in_model_summary=('Ego predictor', 'ego capacity'))

    @property
    def vis_ego_predictor(self) -> int:
        """
        Choose whether to visualize trajectories forecasted by the ego
        predictior.
        It accepts three values:

        - `0`: Do nothing;
        - `1`: Visualize ego predictor's all predictions;
        - `2`: Visualize ego predictor's mean predicton for each neighbor.

        NOTE that this arg only works in the *Playground* mode, or the program
        will be killed immediately.
        """
        return self._arg('vis_ego_predictor', 0, argtype=TEMPORARY)

    def _init_all_args(self):
        super()._init_all_args()

        if ((self.vis_ego_predictor)
                and (self._terminal_args is not None)
                and ('playground' not in ''.join(self._terminal_args))):
            self.log('Arg `vis_ego_predictor` can be only used in the ' +
                     'playground mode!',
                     level='error', raiseError=ValueError)
