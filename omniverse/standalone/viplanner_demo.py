# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

# [18744] IssacSim

import argparse

# omni-isaac-lab
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--model_dir", default=None, type=str, help="Path to model directory.")

# add applauncher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
# [18744] interface with renderer
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.viplanner.viplanner import VIPlannerAlgo
from pxr import UsdGeom


from get_d_star_path import get_d_star_path

"""
Main
"""


def main():
    """Imports all legged robots supported in IsaacLab and applies zero actions."""

    # create environment cfg
    if args_cli.scene == "matterport":
        env_cfg = ViPlannerMatterportCfg(seed=1234)
        goal_pos = torch.tensor([8.0, -13.5, 1.0])
    elif args_cli.scene == "carla":
        env_cfg = ViPlannerCarlaCfg(seed=1234)
        goal_pos = torch.tensor([137, 111.0, 1.0])
    elif args_cli.scene == "warehouse":
        env_cfg = ViPlannerWarehouseCfg(seed=1234)
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # create environment
    # [18744] env : interface to Isaac Sim environment
    env = ManagerBasedRLEnv(env_cfg)

    # adjust the intrinsics of the camera
    depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # Make sure that groundplane is invisible
    if args_cli.scene == "carla":
        assert (
            prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
        )

    # [18744] obs: dictionary contains the raw sensor data from the simulator
    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    # pause the simulator
    # env.sim.pause()

    # load viplanner
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=env.device)

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # initial paths
    _, paths, fear = viplanner.plan_dual(
        obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goals
    )

    # [18744] Fear reaction tracking for stuck detection
    fear_buffer = 0
    buffer_size = 3  # Number of consecutive high-fear frames to trigger reaction
    is_fear_reaction = False

    # Simulate physics
    # [18744] main logic: get sensor data, runs the planner and sends the resulting path as the next action
    while simulation_app.is_running():
        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue
            # [18744] previous action used as the action in this moment for the robot
            # [18744] put action into render (source: ViPlannerMatterportCfg)
            obs = env.step(action=paths.view(paths.shape[0], -1))[0]

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
        if torch.any(
            torch.norm(obs["planner_transform"]["cam_position"] - goals)
            > viplanner.train_config.data_cfg[0].max_goal_distance
        ):
            print(
                f"[WARNING]: Max goal distance is {viplanner.train_config.data_cfg[0].max_goal_distance} but goal is {torch.norm(obs['planner_transform']['cam_position'] - goals)} away from camera position! Please select new goal!"
            )
            env.sim.pause()
            continue
        # [18744] transfer into camera's position
        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # ------------------------------------------------------------------
        # [18744] Accessing Sensor Data
        # ------------------------------------------------------------------
        raw_depth = obs["planner_image"]["depth_measurement"]               # Shape: [Num_Envs, H, W]
        raw_semantic = obs["planner_image"]["semantic_measurement"]         # Shape: [Num_Envs, H, W]
        raw_cam_position = obs["planner_transform"]["cam_position"]         # Shape: [Num_Envs, 3]
        raw_cam_orientation = obs["planner_transform"]["cam_orientation"]   # Shape: [Num_Envs, 4]
        
        # [18744] Run D* Lite Planner
        # Using the first environment's data (index 0) for the demo
        d_lite_path_cam = get_d_star_path(raw_depth[0], goal_cam_frame[0], depth_intrinsic)
        # ------------------------------------------------------------------

        # [18744] run neurak network planner, output path
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
        # [18744] convert waypoints from the camera's frame into the world's coordinate frame
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # ------------------------------------------------------------------
        # [18744] Path Post-Processing for Isaac Sim Demo
        #
        # The `paths` variable here contains the final world-frame waypoints
        # before they are sent to the robot controller in the next loop iteration.
        # Possible place for the CHECKER.
        #
        # Example: paths = checker(paths, d_lit_path)
        
        # [18744] Transform D* Lite path to world frame
        d_lite_path_world = viplanner.path_transformer(
            d_lite_path_cam.unsqueeze(0), raw_cam_position[0:1], raw_cam_orientation[0:1]
        )

        # ------------------------------------------------------------------
        # [18744] Fear Reaction Detection (Stuck/Unsafe Path Detection)
        # ------------------------------------------------------------------
        # Check if the path has high fear (potentially unsafe/stuck)
        fear_value = fear[0].item() if fear.numel() > 0 else 0.0
        
        # Update fear buffer (similar to ROS implementation)
        if fear_value > 0.7:
            fear_buffer = min(fear_buffer + 1, buffer_size + 1)
            print(f"[WARNING]: High fear detected: {fear_value:.3f} (buffer: {fear_buffer}/{buffer_size})")
        else:
            fear_buffer = max(fear_buffer - 1, 0)
        
        # Trigger fear reaction if buffer exceeds threshold
        if fear_buffer > buffer_size:
            if not is_fear_reaction:
                print(f"[STUCK DETECTED]: Fear threshold exceeded! Switching to D* Lite fallback planner.")
                print(f"[STUCK DETECTED]: Fear value: {fear_value:.3f}")
                is_fear_reaction = True
            # [18744] FALLBACK: Use D* Lite path when neural network is uncertain
            paths = d_lite_path_world
            print(f"[FALLBACK]: Using D* Lite path instead of neural network path")
        elif fear_buffer <= 0:
            if is_fear_reaction:
                print(f"[RECOVERY]: Fear subsided, resuming neural network planner.")
                is_fear_reaction = False
        # ------------------------------------------------------------------

        # draw path
        viplanner.debug_draw(paths, fear, goals)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
