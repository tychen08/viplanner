"""
Headless-compatible visualizer using pygame for path visualization.
Captures depth maps, semantic maps, and path data without requiring UDP or Isaac Sim rendering.
"""

import numpy as np
import torch
import pygame
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2
import json
import gzip
import pickle


class HeadlessVisualizer:
    """
    Headless visualizer that renders:
    - Depth map (top-down view)
    - Path positions (neural network planner)
    - D* Lite path positions
    - Goal position
    - Camera position
    """

    def __init__(
        self,
        output_dir: str = "./headless_viz_output",
        window_size: Tuple[int, int] = (1024, 1024),
        depth_scale: float = 1.0,
        save_images: bool = True,
        save_data: bool = True,
        data_compression: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save visualization images and data logs
            window_size: Size of the visualization window (width, height)
            depth_scale: Scale factor for depth map visualization
            save_images: Whether to save images to disk
            save_data: Whether to save data logs (paths, positions, etc.)
            data_compression: Whether to compress data logs with gzip
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "frames"
        self.data_dir = self.output_dir / "data"
        self.images_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.window_size = window_size
        self.depth_scale = depth_scale
        self.save_images = save_images
        self.save_data = save_data
        self.data_compression = data_compression
        self.frame_count = 0
        self.data_log = []  # Store all frame data

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("ViPlanner Headless Visualization")
        self.clock = pygame.time.Clock()

        # Color palette
        self.colors = {
            "background": (30, 30, 30),
            "depth": (100, 100, 100),
            "path": (0, 255, 0),  # Green
            "d_lite_path": (0, 0, 255),  # Blue
            "goal": (255, 0, 0),  # Red
            "camera": (255, 255, 0),  # Yellow
            "text": (255, 255, 255),  # White
        }

    def depth_to_topdown_view(
        self, depth_map: torch.Tensor, intrinsic: torch.Tensor, grid_size: float = 50.0
    ) -> np.ndarray:
        """
        Convert depth map to top-down view (bird's eye view).

        Args:
            depth_map: Depth map [H, W] in meters
            intrinsic: Camera intrinsic matrix [3, 3]
            grid_size: Size of each grid cell in pixels

        Returns:
            Top-down view as numpy array
        """
        depth_np = depth_map.cpu().numpy() if torch.is_tensor(depth_map) else depth_map
        h, w = depth_np.shape

        # Create empty grid
        max_distance = 10.0  # meters
        grid_cells = int(max_distance * 2 / grid_size)
        topdown = np.zeros((grid_cells, grid_cells), dtype=np.uint8)

        # Get camera intrinsics
        fx = intrinsic[0, 0].item() if torch.is_tensor(intrinsic) else intrinsic[0, 0]
        fy = intrinsic[1, 1].item() if torch.is_tensor(intrinsic) else intrinsic[1, 1]
        cx = intrinsic[0, 2].item() if torch.is_tensor(intrinsic) else intrinsic[0, 2]
        cy = intrinsic[1, 2].item() if torch.is_tensor(intrinsic) else intrinsic[1, 2]

        # Convert depth to 3D points and project to top-down view
        for y in range(0, h, 2):  # Skip for efficiency
            for x in range(0, w, 2):
                d = depth_np[y, x]
                if d > 0.1 and d < max_distance:  # Valid depth range
                    # Unproject to 3D
                    z = d
                    x_3d = (x - cx) * z / fx
                    y_3d = (y - cy) * z / fy

                    # Project to top-down view (z=x_3d horizontal, y=-y_3d vertical)
                    grid_x = int((x_3d + max_distance) / grid_size)
                    grid_y = int((max_distance - y_3d) / grid_size)

                    if 0 <= grid_x < grid_cells and 0 <= grid_y < grid_cells:
                        topdown[grid_y, grid_x] = min(255, topdown[grid_y, grid_x] + 2)

        # Normalize to uint8
        topdown = np.clip(topdown, 0, 255).astype(np.uint8)
        return topdown

    def world_to_grid(
        self, world_pos: np.ndarray, grid_size: float = 50.0, max_distance: float = 10.0
    ) -> Tuple[int, int]:
        """
        Convert world position to grid coordinates.

        Args:
            world_pos: Position in world frame [x, y]
            grid_size: Grid cell size in pixels
            max_distance: Maximum distance to visualize

        Returns:
            (grid_x, grid_y) coordinates
        """
        grid_cells = int(max_distance * 2 / grid_size)
        grid_x = int((world_pos[0] + max_distance) / grid_size)
        grid_y = int((max_distance - world_pos[1]) / grid_size)

        # Clamp to grid
        grid_x = max(0, min(grid_x, grid_cells - 1))
        grid_y = max(0, min(grid_y, grid_cells - 1))

        return grid_x, grid_y

    def draw_paths_on_grid(
        self,
        topdown: np.ndarray,
        paths: torch.Tensor,
        d_lite_path: torch.Tensor,
        goal_pos: torch.Tensor,
        camera_pos: torch.Tensor,
        grid_size: float = 50.0,
        max_distance: float = 10.0,
    ) -> np.ndarray:
        """
        Draw paths and key points on top-down grid.

        Args:
            topdown: Base top-down depth view
            paths: Neural network path waypoints [N, 3] in world frame
            d_lite_path: D* Lite path waypoints [N, 3] in world frame
            goal_pos: Goal position [3]
            camera_pos: Camera position [3]
            grid_size: Grid cell size in pixels
            max_distance: Maximum distance to visualize

        Returns:
            Visualization as numpy array [H, W, 3]
        """
        grid_cells = int(max_distance * 2 / grid_size)
        
        # Create RGB image from depth
        viz = cv2.cvtColor(topdown, cv2.COLOR_GRAY2BGR)

        # Extract 2D positions from 3D coordinates
        paths_2d = paths[:, :2].cpu().numpy() if torch.is_tensor(paths) else paths[:, :2]
        d_lite_2d = d_lite_path[:, :2].cpu().numpy() if torch.is_tensor(d_lite_path) else d_lite_path[:, :2]
        goal_2d = goal_pos[:2].cpu().numpy() if torch.is_tensor(goal_pos) else goal_pos[:2]
        camera_2d = camera_pos[:2].cpu().numpy() if torch.is_tensor(camera_pos) else camera_pos[:2]

        # Draw camera position (yellow circle)
        cam_grid = self.world_to_grid(camera_2d, grid_size, max_distance)
        cv2.circle(viz, cam_grid, 5, self.colors["camera"], -1)

        # Draw neural network path (green line)
        for i in range(len(paths_2d) - 1):
            pt1 = self.world_to_grid(paths_2d[i], grid_size, max_distance)
            pt2 = self.world_to_grid(paths_2d[i + 1], grid_size, max_distance)
            cv2.line(viz, pt1, pt2, self.colors["path"], 2)

        # Draw path waypoints
        for pt in paths_2d:
            grid_pt = self.world_to_grid(pt, grid_size, max_distance)
            cv2.circle(viz, grid_pt, 3, self.colors["path"], -1)

        # Draw D* Lite path (blue line)
        for i in range(len(d_lite_2d) - 1):
            pt1 = self.world_to_grid(d_lite_2d[i], grid_size, max_distance)
            pt2 = self.world_to_grid(d_lite_2d[i + 1], grid_size, max_distance)
            cv2.line(viz, pt1, pt2, self.colors["d_lite_path"], 2)

        # Draw D* Lite waypoints
        for pt in d_lite_2d:
            grid_pt = self.world_to_grid(pt, grid_size, max_distance)
            cv2.circle(viz, grid_pt, 3, self.colors["d_lite_path"], -1)

        # Draw goal position (red circle)
        goal_grid = self.world_to_grid(goal_2d, grid_size, max_distance)
        cv2.circle(viz, goal_grid, 8, self.colors["goal"], -1)

        return viz

    def add_text_info(
        self, viz: np.ndarray, fear_value: float, frame_num: int, is_stuck: bool = False
    ) -> np.ndarray:
        """
        Add text information to visualization.

        Args:
            viz: Visualization image
            fear_value: Fear value from planner
            frame_num: Frame number
            is_stuck: Whether stuck detection is triggered

        Returns:
            Visualization with text
        """
        h, w = viz.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = self.colors["text"]

        y_offset = 25
        cv2.putText(
            viz,
            f"Frame: {frame_num}",
            (10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.putText(
            viz,
            f"Fear: {fear_value:.3f}",
            (10, y_offset + 25),
            font,
            font_scale,
            color,
            thickness,
        )

        if is_stuck:
            cv2.putText(
                viz,
                "[STUCK] Using D* Lite Fallback",
                (10, y_offset + 50),
                font,
                font_scale + 0.2,
                self.colors["d_lite_path"],
                2,
            )

        # Add legend
        legend_y = h - 100
        cv2.putText(
            viz,
            "Legend:",
            (10, legend_y),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.line(viz, (10, legend_y + 15), (40, legend_y + 15), self.colors["path"], 2)
        cv2.putText(
            viz,
            "Neural Path",
            (45, legend_y + 20),
            font,
            font_scale * 0.8,
            color,
            thickness,
        )

        cv2.line(viz, (10, legend_y + 40), (40, legend_y + 40), self.colors["d_lite_path"], 2)
        cv2.putText(
            viz,
            "D* Lite Path",
            (45, legend_y + 45),
            font,
            font_scale * 0.8,
            color,
            thickness,
        )

        cv2.circle(viz, (25, legend_y + 65), 5, self.colors["goal"], -1)
        cv2.putText(
            viz,
            "Goal",
            (45, legend_y + 70),
            font,
            font_scale * 0.8,
            color,
            thickness,
        )

        return viz

    def visualize_step(
        self,
        depth_map: torch.Tensor,
        paths: torch.Tensor,
        d_lite_path: torch.Tensor,
        goal_pos: torch.Tensor,
        camera_pos: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        fear_value: float = 0.0,
        is_stuck: bool = False,
        display: bool = False,
    ) -> np.ndarray:
        """
        Create and optionally display visualization for a single step.

        Args:
            depth_map: Depth map [H, W]
            paths: Neural network path [N, 3]
            d_lite_path: D* Lite path [N, 3]
            goal_pos: Goal position [3]
            camera_pos: Camera position [3]
            camera_intrinsic: Camera intrinsic matrix [3, 3]
            fear_value: Fear value from planner
            is_stuck: Whether stuck detection triggered
            display: Whether to display using pygame

        Returns:
            Visualization as numpy array [H, W, 3]
        """
        # Create top-down view from depth
        topdown = self.depth_to_topdown_view(depth_map, camera_intrinsic, grid_size=50.0)

        # Draw paths on grid
        viz = self.draw_paths_on_grid(
            topdown,
            paths,
            d_lite_path,
            goal_pos,
            camera_pos,
            grid_size=50.0,
        )

        # Add text information
        viz = self.add_text_info(viz, fear_value, self.frame_count, is_stuck)

        # Prepare data for logging
        frame_data = {
            "timestamp": self.frame_count,
            "fear_value": float(fear_value),
            "is_stuck": bool(is_stuck),
            "neural_path": paths.cpu().numpy().tolist() if torch.is_tensor(paths) else paths.tolist(),
            "d_lite_path": d_lite_path.cpu().numpy().tolist() if torch.is_tensor(d_lite_path) else d_lite_path.tolist(),
            "goal_position": goal_pos.cpu().numpy().tolist() if torch.is_tensor(goal_pos) else goal_pos.tolist(),
            "camera_position": camera_pos.cpu().numpy().tolist() if torch.is_tensor(camera_pos) else camera_pos.tolist(),
        }

        # Save frame data to log
        self.save_frame_data(frame_data, depth_map=depth_map, save_depth=False)

        # Save to disk if enabled
        if self.save_images:
            output_path = self.images_dir / f"frame_{self.frame_count:06d}.png"
            cv2.imwrite(str(output_path), viz)

        # Display with pygame if requested
        if display:
            self._display_pygame(viz)

        self.frame_count += 1
        return viz

    def _display_pygame(self, image: np.ndarray):
        """Display image using pygame."""
        # Convert BGR to RGB for pygame
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to window size if needed
        if image_rgb.shape[:2] != self.window_size:
            image_rgb = cv2.resize(image_rgb, self.window_size)

        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        self.clock.tick(30)  # 30 FPS
        return True

    def get_path_data(
        self, paths: torch.Tensor, d_lite_path: torch.Tensor, goal_pos: torch.Tensor, camera_pos: torch.Tensor
    ) -> dict:
        """
        Extract path position data as dictionary (useful for logging/analysis).

        Returns:
            Dictionary with path positions
        """
        return {
            "neural_path": paths.cpu().numpy().tolist() if torch.is_tensor(paths) else paths.tolist(),
            "d_lite_path": d_lite_path.cpu().numpy().tolist() if torch.is_tensor(d_lite_path) else d_lite_path.tolist(),
            "goal_position": goal_pos.cpu().numpy().tolist() if torch.is_tensor(goal_pos) else goal_pos.tolist(),
            "camera_position": camera_pos.cpu().numpy().tolist() if torch.is_tensor(camera_pos) else camera_pos.tolist(),
            "frame": self.frame_count,
        }

    def save_frame_data(
        self,
        frame_data: Dict[str, Any],
        depth_map: Optional[torch.Tensor] = None,
        save_depth: bool = False,
    ) -> None:
        """
        Save frame data (paths, positions) to log.

        Args:
            frame_data: Dictionary with path and position data
            depth_map: Optional depth map to save
            save_depth: Whether to save depth map (as compressed numpy)
        """
        if not self.save_data:
            return

        frame_entry = {
            "frame": self.frame_count,
            "data": frame_data,
        }

        # Optionally save depth map
        if save_depth and depth_map is not None:
            depth_np = depth_map.cpu().numpy() if torch.is_tensor(depth_map) else depth_map
            depth_path = self.data_dir / f"depth_{self.frame_count:06d}.npz"
            np.savez_compressed(str(depth_path), depth=depth_np)
            frame_entry["depth_file"] = str(depth_path.relative_to(self.output_dir))

        self.data_log.append(frame_entry)

    def save_data_log(self, name: str = "data_log") -> Path:
        """
        Save accumulated frame data to disk.

        Args:
            name: Name of the log file (without extension)

        Returns:
            Path to saved log file
        """
        if not self.data_log:
            print("[Warning] No data to save")
            return None

        # Save as JSON for easy inspection
        json_path = self.data_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(self.data_log, f, indent=2)

        # Also save as compressed pickle for efficiency
        if self.data_compression:
            pickle_path = self.data_dir / f"{name}.pkl.gz"
            with gzip.open(pickle_path, "wb") as f:
                pickle.dump(self.data_log, f)
            print(f"[Saved] Data log (compressed): {pickle_path}")
        else:
            pickle_path = self.data_dir / f"{name}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(self.data_log, f)
            print(f"[Saved] Data log: {pickle_path}")

        print(f"[Saved] Data log (JSON): {json_path}")
        return json_path

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save metadata about the recording session.

        Args:
            metadata: Dictionary with session information

        Returns:
            Path to metadata file
        """
        metadata_path = self.output_dir / "metadata.json"
        metadata["total_frames"] = self.frame_count
        metadata["num_frames_with_data"] = len(self.data_log)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Saved] Metadata: {metadata_path}")
        return metadata_path

    def close(self):
        """Close pygame display and save data logs."""
        # Save data before closing
        if self.save_data and self.data_log:
            self.save_data_log("data_log")

        pygame.quit()


if __name__ == "__main__":
    print("HeadlessVisualizer module loaded successfully.")
