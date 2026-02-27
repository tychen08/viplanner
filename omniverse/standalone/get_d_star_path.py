import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../d-star-lite")))
try:
    from grid import GridWorld
    from d_star_lite import initDStarLite, scanForObstacles, computeShortestPath, nextInShortestPath, stateNameToCoords
except ImportError:
    print("[WARNING] Could not import d-star-lite library. Please check the path.")


def get_d_star_path(depth_tensor, goal_cam, intrinsics):
    """
    Generates a path using D* Lite based on depth sensor data.
    depth_tensor: [H, W] tensor on GPU/CPU
    goal_cam: [3] tensor (x, y, z) in camera frame
    intrinsics: [3, 3] tensor camera intrinsics
    """
    # Grid Configuration
    X_DIM, Y_DIM = 40, 40  # 40x40 grid
    CELL_RES = 0.25        # 0.25m per cell -> 10m x 10m area
    
    # Initialize GridWorld
    graph = GridWorld(X_DIM, Y_DIM)
    
    # Map Depth to Obstacles (Simplified Downsampling)
    d_np = depth_tensor.cpu().numpy()
    H, W = d_np.shape
    fx, cx = intrinsics[0, 0].item(), intrinsics[0, 2].item()
    
    # Sparse sampling for speed
    step = 8 
    for v in range(0, H, step):
        for u in range(0, W, step):
            z = d_np[v, u]
            if z <= 0.1 or z > 9.0: continue # Ignore invalid/far points
            
            # Project to Camera Frame (X right, Z forward)
            x = (u - cx) * z / fx
            
            # Map to Grid (Robot at X=Center, Y=0)
            # Grid X -> World X (Right), Grid Y -> World Z (Forward)
            grid_x = int(x / CELL_RES + X_DIM / 2)
            grid_y = int(z / CELL_RES)
            
            if 0 <= grid_x < X_DIM and 0 <= grid_y < Y_DIM:
                graph.cells[grid_y][grid_x] = -1 # Mark obstacle

    # Setup Start and Goal
    s_start = f"x{int(X_DIM/2)}y0"
    gx = int(goal_cam[0].item() / CELL_RES + X_DIM / 2)
    gy = int(goal_cam[2].item() / CELL_RES)
    gx = max(0, min(X_DIM - 1, gx))
    gy = max(0, min(Y_DIM - 1, gy))
    s_goal = f"x{gx}y{gy}"
    
    graph.setStart(s_start)
    graph.setGoal(s_goal)
    
    # Run D* Lite
    queue = []
    k_m = 0
    # 1. Init (plans on empty grid)
    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)
    # 2. Scan (updates edges based on cells we filled)
    scanForObstacles(graph, queue, s_start, 100, k_m) # Large range to cover grid
    # 3. Replan
    computeShortestPath(graph, queue, s_start, k_m)
    
    # Extract Path
    path_points = []
    curr = s_start
    for _ in range(100): # Limit steps
        if curr == s_goal: break
        try:
            curr = nextInShortestPath(graph, curr)
            coords = stateNameToCoords(curr)
            # Convert Grid -> Camera Frame
            x_w = (coords[0] - X_DIM / 2) * CELL_RES
            z_w = coords[1] * CELL_RES
            path_points.append([x_w, 0.0, z_w])
        except:
            break
    
    if not path_points:
        return torch.zeros((1, 3), device=depth_tensor.device)
    return torch.tensor(path_points, device=depth_tensor.device)