import json
import numpy as np

def extract_trajectory_frames(json_file):
    """
    Reads the JSON file containing camera poses, sorts them by frame name 
    to ensure chronological order, and returns a list of 4x4 pose matrices 
    as NumPy arrays.
    """
    with open(json_file, "r") as file:
        trajectory_data = json.load(file)

    # Extract the frame names and sort them
    frame_names = list(trajectory_data.keys())
    frame_names.sort()
    
    # Create an ordered list of pose matrices
    poses = [np.asarray(trajectory_data[name], dtype=float) for name in frame_names]
    
    return poses


def filter_trajectory_frames(poses, interval=15):
    """
    Selects every 15th trajectory frame (or whatever interval is specified) 
    from a list of poses and returns the filtered list.
    """
    # Use standard Python list slicing to get every 'interval' element
    return poses[::interval]


def compute_trajectory_deltas(poses, working_distance=25.0):
    """
    Calculates the 2D apparent displacement and direction between consecutive camera poses.
    
    By projecting a virtual tissue landmark at distance `working_distance` (in mm)
    along the optical axis of frame A into the camera view of frame B, this method
    incorporates both linear translation (dx, dy) and camera angular tilt (pitch, yaw).
    
    Returns a dictionary containing:
    - 'dx': Array of effective X displacements (translation + tilt)
    - 'dy': Array of effective Y displacements (translation + tilt)
    - 'dz': Array of local Z translations (zoom into/out of cavity)
    - 'magnitudes': Array of 2D panning magnitudes (sqrt(dx^2 + dy^2))
    - 'angles': Array of directional angles in radians (arctan2(dy, dx))
    - 'dx_pure': Array of pure linear X translations without rotation
    - 'dy_pure': Array of pure linear Y translations without rotation
    """
    effective_deltas = []
    pure_deltas = []
    
    for i in range(len(poses) - 1):
        pose_A = poses[i]
        pose_B = poses[i + 1]
        
        # 1. Extract translation (world positions) and rotation matrices
        t_A = pose_A[:3, 3]
        t_B = pose_B[:3, 3]
        R_A = pose_A[:3, :3]
        R_B = pose_B[:3, :3]
        
        # 2. Pure linear translation in Camera A local frame
        delta_world = t_B - t_A
        delta_local_pure = np.dot(R_A.T, delta_world)
        pure_deltas.append(delta_local_pure)
        
        # 3. Virtual tissue landmark at distance Z along optical axis of Camera A:
        # P_world = t_A + R_A * [0, 0, Z] = t_A + R_A[:, 2] * Z
        P_world = t_A + R_A[:, 2] * working_distance
        
        # 4. Project that landmark into Camera B coordinate frame: P_B = R_B^T * (P_world - t_B)
        P_B = np.dot(R_B.T, P_world - t_B)
        
        # Apparent motion seen on the camera sensor:
        dx_eff = -P_B[0]
        dy_eff = -P_B[1]
        dz_eff = delta_local_pure[2]
        
        effective_deltas.append([dx_eff, dy_eff, dz_eff])
        
    effective_deltas = np.array(effective_deltas)
    pure_deltas = np.array(pure_deltas)
    
    if len(effective_deltas) > 0:
        dx = effective_deltas[:, 0]
        dy = effective_deltas[:, 1]
        dz = effective_deltas[:, 2]
        dx_pure = pure_deltas[:, 0]
        dy_pure = pure_deltas[:, 1]
    else:
        dx, dy, dz = np.array([]), np.array([]), np.array([])
        dx_pure, dy_pure = np.array([]), np.array([])
        
    magnitudes = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    
    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "magnitudes": magnitudes,
        "angles": angles,
        "dx_pure": dx_pure,
        "dy_pure": dy_pure
    }

