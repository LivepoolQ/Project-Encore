"""
@Author: Ziqian Zou
@Date: 2026-02-05 16:10:21
@LastEditors: Ziqian Zou
@LastEditTime: 2026-02-05 16:32:28
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""

import numpy as np
import qpid.mods.vis.helpers
from qpid.mods.vis.helpers.__normal2D import ADD
from qpid.utils import get_relative_path

def vis(self, source: np.ndarray,
            obs: np.ndarray | None = None,
            gt: np.ndarray | None = None,
            pred: np.ndarray | None = None,
            neighbor: np.ndarray | None = None,
            background: np.ndarray | None = None,
            pred_colors: np.ndarray | None = None,
            *args, **kwargs):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param source: The image file.
        :param obs: (optional) The observations in *pixel* scale.
        :param gt: (optional) The ground truth in *pixel* scale.
        :param pred: (optional) The predictions in *pixel* scale,\
            shape = `(K, steps, dim)`.
        :param neighbor: (optional) The observed neighbors' positions\
             in *pixel* scale, shape = `(batch, dim)`.
        :param draw_distribution: Controls whether to draw as a distribution.
        :param alpha: The alpha channel coefficient.
        """
        f = np.zeros([source.shape[0], source.shape[1], 4])

        # draw neighbors' observed trajectories
        if neighbor is not None:
            f = self.helper.draw_traj(f, neighbor[..., -1:, :],
                                      self.current_file,
                                      do_not_draw_lines=True)

            neighbor = neighbor if self.vis_args.draw_full_neighbors \
                else neighbor[..., -1:, :]
            for nei in neighbor:
                f = self.helper.draw_traj(f, nei, self.neighbor_file)

        # draw predicted trajectories
        if pred is not None:
            if self.vis_args.draw_distribution:
                alpha = 0.8 if not self.vis_args.draw_on_empty_canvas else 1.0
                f = self.helper.draw_dis(f, pred, alpha=alpha,
                                         steps=self.vis_args.distribution_steps)
            else:
                # if pred_colors is None:
                #     pred_colors = 255 * np.random.rand(pred.shape[0], 3)

                # for (pred_k, color_k) in zip(pred, pred_colors):
                #     f = self.helper.draw_traj(
                #         f, pred_k, self.pred_file,
                #         color=color_k)
                for pred_k in pred:
                    f = self.helper.draw_traj(f, pred_k, self.pred_file)

        # draw observed and groundtruth trajectories
        if obs is not None:
            if obs.ndim == 2:
                obs = obs[np.newaxis]

            for _obs in obs:
                f = self.helper.draw_traj(f, _obs, self.obs_file)

        if gt is not None:
            if gt.ndim == 2:
                gt = gt[np.newaxis]

            for _gt in gt:
                f = self.helper.draw_traj(f, _gt, self.gt_file)

        # draw the background image
        if background is not None:
            f = ADD(background, f, [f.shape[1]//2, f.shape[0]//2])

        # add the original image
        f = ADD(source, f, [f.shape[1]//2, f.shape[0]//2])
        return f

def modify_qpid_utils(mod_vis_func: bool|int, mod_pred_img: bool|int):
    if mod_vis_func:
        qpid.mods.vis.helpers.Normal2DCanvas.vis = vis
    if mod_pred_img:
        qpid.mods.vis.helpers.__normal2D.PRED_IMAGE = get_relative_path(__file__, 'group_member.png')