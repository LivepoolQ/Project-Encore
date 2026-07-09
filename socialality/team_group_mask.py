"""
@Author: Ziqian Zou
@Date: 2026-07-08 17:53:43
@LastEditors: Ziqian Zou
@LastEditTime: 2026-07-09 09:36:54
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""
import csv
import os

import numpy as np
import torch

from qpid.base import BaseManager
from qpid.dataset.__base import BaseExtInputManager


class TeamGroupMaskManager(BaseExtInputManager):
    TEMP_FILE = 'team_group_mask.npy'
    INPUT_TYPE = 'TEAM_GROUP_MASK'

    def __init__(self, manager: BaseManager, name='Team Group Mask Manager'):
        super().__init__(manager, name)

    def save(self, trajs: np.ndarray, agents: list, *args, **kwargs):
        # 1. Parse ann.csv to get coordinates and teams for each frame
        frame_coords = {}
        with open(self.working_clip.annpath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    frame_id = int(row[0])
                    x = float(row[2])
                    y = float(row[3])
                    team = row[4]
                    if frame_id not in frame_coords:
                        frame_coords[frame_id] = []
                    frame_coords[frame_id].append((x, y, team))

        # 2. For each agent, calculate the teammate mask for its neighbors
        max_agents = self.args.max_agents
        group_masks = []
        
        for agent in agents:
            ego_team = agent._type
            obs_len = agent.obs_length
            last_frame_id = agent._frames[obs_len - 1]
            
            ref_pos = agent.traj[-1]  # Shape: (2,)
            nei_rel_traj = agent.traj_neighbor  # Shape: (max_agents, obs, dim)
            nei_rel_pos = nei_rel_traj[:, obs_len - 1, :]  # Shape: (max_agents, 2)
            nei_abs_pos = nei_rel_pos + ref_pos  # Shape: (max_agents, 2)
            
            mask = np.zeros(max_agents, dtype=np.float32)
            
            if ego_team != "Ball":
                for j in range(max_agents):
                    nx, ny = nei_abs_pos[j]
                    
                    # If this is a padding position, skip
                    if abs(nx) > 1000000 or abs(ny) > 1000000:
                        continue
                        
                    # Find closest player in this frame
                    best_team = "Ball"
                    min_dist = float('inf')
                    for px, py, pteam in frame_coords.get(last_frame_id, []):
                        dist = (px - nx)**2 + (py - ny)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_team = pteam
                            
                    # If it's a valid player match on the same team, mark as teammate
                    if min_dist < 0.2 and best_team == ego_team and best_team != "Ball":
                        mask[j] = 1.0
                        
            group_masks.append(mask)

        if not self.temp_file:
            raise ValueError
        os.makedirs(self.temp_dir, exist_ok=True)
        np.save(self.temp_file, np.array(group_masks, dtype=np.float32))

    def load(self, *args, **kwargs):
        if not self.temp_file:
            raise ValueError
        return np.load(self.temp_file)

def team_group_mask_handler(manager):
    # Ensure only one instance of TeamGroupMaskManager is added
    for mgr in manager.ext_mgrs:
        if mgr.INPUT_TYPE == 'TEAM_GROUP_MASK':
            return
    manager.ext_mgrs.append(TeamGroupMaskManager(manager))
