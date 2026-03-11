# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path, read_file
from omni.isaac.lab.utils.math import quat_rotate_inverse


# -- Navigation Action
class NavigationAction(ActionTerm):
    """Actions to navigate a robot by following some path."""

    cfg: NavigationActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: NavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # check if policy file exists
        self.low_level_policy = None
        if self.cfg.low_level_policy_file is not None:
            if not check_file_path(self.cfg.low_level_policy_file):
                raise FileNotFoundError(f"Policy file '{self.cfg.low_level_policy_file}' does not exist.")
            file_bytes = read_file(self.cfg.low_level_policy_file)
            # load policies
            self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
            self.low_level_policy = torch.jit.freeze(self.low_level_policy.eval())

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = (
            self.cfg.path_length * 3
        )  # [vx, vy, omega] --> vx: [-0.5,1.0], vy: [-0.5,0.5], omega: [-1.0,1.0]
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(
            (self.num_envs, self.cfg.path_length, 3), device=self.device
        )

        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt

        self._counter = 0
        self._debug_counter = 0 # 加入獨立的 debug counter 以便正確印出 log

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process low-level navigation actions. This function is called with a frequency of 10Hz"""
        # Store low level navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        # reshape into 3D path
        self._processed_navigation_velocity_actions[:] = actions.clone().view(self.num_envs, self.cfg.path_length, 3)

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            # -- update command
            self._env.command_manager.compute(dt=self._low_level_step_dt)
            
            # 1. Get Robot Pose
            robot = self._env.scene["robot"]
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w

            # 2. Get Target Point (Dynamic Lookahead - 修正版)
            path_points = self._processed_navigation_velocity_actions[0] # [Path_Len, 3]
            
            # 算出所有路徑點與車子的距離
            dists = torch.norm(path_points[:, :2] - root_pos_w[0, :2], dim=1)
            
            # 找到離車子「最近」的點的 Index (避免被過去的點綁架)
            closest_idx = torch.argmin(dists)
            
            # 只從最近的點往後看，找第一個距離大於 0.8m 的點
            lookahead_dists = dists[closest_idx:]
            valid_indices = torch.nonzero(lookahead_dists > 0.8, as_tuple=True)[0]
            
            if len(valid_indices) > 0:
                target_idx = closest_idx + valid_indices[0] 
                target_pos_w_single = path_points[target_idx, :2]
            else:
                target_pos_w_single = path_points[-1, :2]
                
            # 擴充維度以符合 batched tensor (形狀: [num_envs, 2])
            target_pos_w = target_pos_w_single.unsqueeze(0).expand(self.num_envs, -1)

            # 3. Compute Error in Robot Frame
            target_vec_w = target_pos_w - root_pos_w[:, :2]
            target_vec_3d_w = torch.cat([target_vec_w, torch.zeros((self.num_envs, 1), device=self.device)], dim=1)
            target_vec_b = quat_rotate_inverse(root_quat_w, target_vec_3d_w)
            
            x_err = target_vec_b[:, 0]
            y_err = target_vec_b[:, 1]

            # 4. Compute Velocities (Turn-Then-Drive Controller)
            angle_error = torch.atan2(y_err, x_err)
            
            # 計算距離最終目標點 (終點) 還有多遠
            dist_to_final_goal = torch.norm(path_points[-1, :2] - root_pos_w[0, :2])
            
            if dist_to_final_goal < 0.3:
                # [抵達煞車機制] 離終點不到 0.3m，直接煞車！
                v_cmd = torch.zeros_like(x_err)
                omega_cmd = torch.zeros_like(angle_error)
                if self._debug_counter % 20 == 0:
                    print(f"[NavAction] 🎉 抵達終點 (距離: {dist_to_final_goal:.2f}m)，煞車！")
            else:
                # [正常追蹤機制]
                v_raw = torch.clamp(2.0 * x_err, 0.0, 1.0) 
                w_raw = torch.clamp(5.0 * angle_error, -2.0, 2.0)
                
                # [Modified for Car/Forklift] Disable turn-in-place
                # Cars cannot turn without moving, so we must allow forward motion even if angle is large.
                # turn_in_place = torch.abs(angle_error) > 0.5
                # v_cmd = torch.where(turn_in_place, torch.zeros_like(v_raw), v_raw)
                v_cmd = v_raw
                omega_cmd = w_raw
                
                if self._debug_counter % 20 == 0:
                    print(f"[NavAction] Robot: {root_pos_w[0, :2].cpu().numpy()} -> Target: {target_pos_w_single.cpu().numpy()}")
                    print(f"            Error: x={x_err[0]:.2f}, y={y_err[0]:.2f} | Cmd: v={v_cmd[0]:.2f}, w={omega_cmd[0]:.2f}")

            self._low_level_actions[:, 0] = v_cmd
            self._low_level_actions[:, 1] = omega_cmd

            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)
            self._debug_counter += 1

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class NavigationActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationAction
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    low_level_policy_file: str | None = None
    """Path to the low level policy file."""
    path_length: int = 51
    """Length of the path to be followed."""


class LimoDiffDriveAction(ActionTerm):
    """Action term for a differential drive robot (e.g. Limo)."""
    cfg: LimoDiffDriveActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: LimoDiffDriveActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.left_joint_ids, _ = self.robot.find_joints(cfg.left_wheel_joint_names)
        self.right_joint_ids, _ = self.robot.find_joints(cfg.right_wheel_joint_names)
        self._action_dim = 2
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._joint_vel_targets = torch.zeros_like(self.robot.data.joint_vel_target)

    @property
    def action_dim(self) -> int: return self._action_dim
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        v, w = actions[:, 0], actions[:, 1]
        omega_left = (v - w * self.cfg.track_width / 2.0) / self.cfg.wheel_radius
        omega_right = (v + w * self.cfg.track_width / 2.0) / self.cfg.wheel_radius
        self._joint_vel_targets[:] = 0.0
        for idx in self.left_joint_ids: self._joint_vel_targets[:, idx] = omega_left
        for idx in self.right_joint_ids: self._joint_vel_targets[:, idx] = omega_right

    def apply_actions(self):
        self.robot.set_joint_velocity_target(self._joint_vel_targets)


@configclass
class LimoDiffDriveActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LimoDiffDriveAction
    asset_name: str = "robot"
    left_wheel_joint_names: list[str] = MISSING
    right_wheel_joint_names: list[str] = MISSING
    wheel_radius: float = MISSING
    track_width: float = MISSING


class AckermannAction(ActionTerm):
    """Action term for an Ackermann drive robot (e.g. Car/Forklift)."""
    cfg: AckermannActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: AckermannActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        
        # Find joints
        self.drive_joint_ids, _ = self.robot.find_joints(cfg.drive_joint_names)
        self.steering_joint_ids, _ = self.robot.find_joints(cfg.steering_joint_names)
        
        self._action_dim = 2 # [v, steering]
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        
        # Buffers for targets
        self._joint_vel_targets = torch.zeros_like(self.robot.data.joint_vel_target)
        self._joint_pos_targets = torch.zeros_like(self.robot.data.joint_pos_target)

    @property
    def action_dim(self) -> int: return self._action_dim
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        v_cmd = actions[:, 0]
        steering_cmd = actions[:, 1]
        
        # 1. Drive Control (Velocity) -> Convert m/s to rad/s
        wheel_vel = v_cmd / self.cfg.wheel_radius
        
        # 2. Steering Control (Position) -> Clamp angle
        steering_pos = torch.clamp(steering_cmd, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        
        self._joint_vel_targets[:] = 0.0
        for idx in self.drive_joint_ids: self._joint_vel_targets[:, idx] = wheel_vel
        for idx in self.steering_joint_ids: self._joint_pos_targets[:, idx] = steering_pos

    def apply_actions(self):
        # [FIX] Slice the tensor to match the specific joint IDs
        self.robot.set_joint_velocity_target(self._joint_vel_targets[:, self.drive_joint_ids], joint_ids=self.drive_joint_ids)
        self.robot.set_joint_position_target(self._joint_pos_targets[:, self.steering_joint_ids], joint_ids=self.steering_joint_ids)

@configclass
class AckermannActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AckermannAction
    asset_name: str = "robot"
    drive_joint_names: list[str] = MISSING
    steering_joint_names: list[str] = MISSING
    wheel_radius: float = 0.1
    max_steering_angle: float = 0.6 # rad (~35 degrees)