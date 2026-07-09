"""
@Author: Ziqian Zou
@Date: 2025-12-11 17:21:42
@LastEditors: Ziqian Zou
@LastEditTime: 2026-07-09 09:34:45
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2025 Ziqian Zou, All Rights Reserved.
"""
import numpy as np

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class SocialalityArgs(EmptyArgs):

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
        - `linear`: Linearly fit the observed trajectory
        - `fc`: Fully connected layer
        - `tran`: Transformer
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
        - `0`: Original GPCC Model ;
        - `1`: Socialality Model.
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
    def adaptive_fov(self) -> int:
        """
        Whether using adaptive fov in perception mechanism.
        """
        return self._arg('adaptive_fov', 0, STATIC,
                         desc_in_model_summary=('use adaptive fov'))
    
    # ---------------------
    # MARK: - Ablation Args
    # ---------------------
    @property
    def use_mixed_trajectory(self) -> int:
        """
        Choose whether to use mixed time window trajectory.
        """
        return self._arg('use_mixed_trajectory', 1, argtype=STATIC, desc_in_model_summary='use mixed trajectory', short_name='mix')
    
    @property
    def fix_distance_anchor(self) -> int:
        """
        Choose whether to fix distance anchor.
        """
        return self._arg('fix_distance_anchor', 0, argtype=STATIC, desc_in_model_summary='fix distance anchor', short_name='fix_dis')
    
    @property
    def fix_speed_anchor(self) -> int:
        """
        Choose whether to speed distance anchor.
        """
        return self._arg('fix_speed_anchor', 0, argtype=STATIC, desc_in_model_summary='fix speed anchor', short_name='fix_speed')
    
    @property
    def set_anchor_value(self) -> int:
        """
        Choose whether to set anchor value globally.
        """
        return self._arg('set_anchor_value', 0, argtype=STATIC, desc_in_model_summary='set anchor value')
    
    @property
    def set_distance_anchor(self) -> float:
        """
        Set distance anchor value globally.
        """
        return self._arg('set_distance_anchor', -1, argtype=STATIC, short_name='set_dis')
    
    @property
    def set_speed_anchor(self) -> float:
        """
        Set distance anchor value globally.
        """
        return self._arg('set_speed_anchor', -1, argtype=STATIC, short_name='set_speed')
    
    @property
    def disable_distance_anchor(self) -> int:
        """
        Choose whether to disable distance anchor.
        """
        return self._arg('disable_distance_anchor', 0, argtype=STATIC, desc_in_model_summary='disable distance anchor', short_name='disable_dis')
    
    @property
    def disable_speed_anchor(self) -> int:
        """
        Choose whether to disable speed anchor.
        """
        return self._arg('disable_speed_anchor', 0, argtype=STATIC, desc_in_model_summary='disable speed anchor', short_name='disable_speed')
    
    @property
    def previews_only(self) -> int:
        """
        Choose whether to only use previews when grouping.
        NOTE This args can only be used when `--use_mixed_trajectory 1`.
        """
        return self._arg('previews_only', 0, argtype=STATIC, desc_in_model_summary='only previews')
    
    @property
    def current_only(self) -> int:
        """
        Choose whether to only use current step when grouping.
        """
        return self._arg('current_only', 0, argtype=STATIC, desc_in_model_summary='only current')
    
    @property
    def set_grouping_ratio(self) -> float:
        """
        Set grouping ratio value, which will randomly assign part of the neighbors as a group.
        NOTE This args can only be used as counterfactual anynasis.
        """
        return self._arg('set_grouping_ratio', -1.0, argtype=TEMPORARY)    
    # --------------------------
    # MARK: - Visualization Args
    # --------------------------
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
    
    @property
    def vis_group_members(self) -> int:
        """
        Choose whether to visualize group members.

        NOTE that this arg only works in the *Playground* mode, or the program
        will be killed immediately.
        """
        return self._arg('vis_group_members', 0, argtype=TEMPORARY)
    
    @property
    def vis_grouping_window(self) -> int:
        """
        Choose whether to visualize grouping window.

        NOTE that this arg only works in the *Playground* mode, or the program
        will be killed immediately. 
        NOTE that this arg only works when the arg `vis_group_members` is
        activated. 
        """
        return self._arg('vis_grouping_window', 0, argtype=TEMPORARY)
    
    @property
    def vis_anchors(self) -> int:
        """
        Choose whether to visualize anchors.
        """
        return self._arg('vis_anchors', 0, argtype=TEMPORARY)

    @property
    def use_team_group_mask(self) -> int:
        """
        Choose whether to use the ground truth team group mask instead of the grouping kernel.

        NOTE This arg can only be used in NBA dataset.
        """
        return self._arg('use_team_group_mask', 0, argtype=STATIC)

    def _init_all_args(self):
        super()._init_all_args()

        if ((self.vis_ego_predictor)
                and (self.vis_group_members)
                and (self._terminal_args is not None)
                and ('playground' not in ''.join(self._terminal_args))):
            self.log('Arg `vis_ego_predictor` can be only used in the ' +
                     'playground mode!',
                     level='error', raiseError=ValueError)
