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


def filter_trajectory_by_distance(poses, distance_threshold=2.0, max_interval=40):
    """
    Selects keyframes based on physical 2D travel distance (in mm) in the camera frame
    rather than fixed frame intervals. This ensures a high signal-to-noise ratio for
    directional angle calculations by avoiding sampling during stationary pauses.
    
    Parameters:
    - poses: List of 4x4 camera pose matrices
    - distance_threshold: Minimum lateral pan distance (in mm) required to trigger a new keyframe (default: 2.0 mm)
    - max_interval: Maximum allowable frames before forcing a keyframe (default: 40 frames)
    
    Returns:
    - filtered_poses: List of 4x4 pose matrices at selected keyframes
    - keyframe_indices: List of integer frame indices for selected keyframes
    """
    keyframe_indices = [0]
    filtered_poses = [poses[0]]
    last_pose = poses[0]
    
    for i in range(1, len(poses)):
        t_last = last_pose[:3, 3]
        t_curr = poses[i][:3, 3]
        R_last = last_pose[:3, :3]
        
        # Calculate displacement in the local camera frame of the last keyframe
        delta_world = t_curr - t_last
        dx = np.dot(R_last[:, 0], delta_world)
        dy = np.dot(R_last[:, 1], delta_world)
        pan_distance = np.sqrt(dx**2 + dy**2)
        
        frames_elapsed = i - keyframe_indices[-1]
        
        # Trigger new keyframe if distance threshold is reached or max_interval exceeded
        if pan_distance >= distance_threshold or frames_elapsed >= max_interval:
            keyframe_indices.append(i)
            filtered_poses.append(poses[i])
            last_pose = poses[i]
            
    return filtered_poses, keyframe_indices


def filter_trajectory_by_indices(poses, keyframe_indices):
    """
    Filters a list of camera poses using specific keyframe frame indices
    (e.g., those selected by video optical flow).
    
    Parameters:
    - poses: List of 4x4 camera pose matrices
    - keyframe_indices: List of integer frame numbers to sample
    
    Returns:
    - filtered_poses: List of 4x4 pose matrices at selected indices
    """
    return [poses[i] for i in keyframe_indices if i < len(poses)]


def compute_trajectory_deltas(poses, working_distance=25.0, use_working_distance=True):
    """
    Calculates the 2D apparent displacement and direction between consecutive camera poses.
    
    Parameters:
    - poses: List of 4x4 camera pose matrices
    - working_distance: Distance (in mm) along optical axis to virtual tissue landmark (default: 25.0 mm)
    - use_working_distance: Boolean flag (default True).
        When True: Projects a virtual tissue landmark at `working_distance` along the optical
                   axis, incorporating both linear translation (dx, dy) and camera angular
                   tilt (pitch, yaw).
        When False: Computes pure linear translation in the camera coordinate frame,
                    ignoring rotational tilt (the baseline behavior before this feature).
    
    Returns a dictionary containing:
    - 'dx': Array of X displacements (effective if use_working_distance=True, else pure translation)
    - 'dy': Array of Y displacements (effective if use_working_distance=True, else pure translation)
    - 'dz': Array of local Z translations (zoom into/out of cavity)
    - 'magnitudes': Array of 2D panning magnitudes (sqrt(dx^2 + dy^2))
    - 'angles': Array of directional angles in radians (arctan2(dy, dx))
    - 'dx_pure': Array of pure linear X translations without rotation
    - 'dy_pure': Array of pure linear Y translations without rotation
    """
    selected_deltas = []
    pure_deltas = []
    
    apply_wd = bool(use_working_distance and working_distance is not None)
    
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
        
        if apply_wd:
            # 3. Virtual tissue landmark at distance Z along optical axis of Camera A:
            # P_world = t_A + R_A * [0, 0, Z] = t_A + R_A[:, 2] * Z
            P_world = t_A + R_A[:, 2] * working_distance
            
            # 4. Project that landmark into Camera B coordinate frame: P_B = R_B^T * (P_world - t_B)
            P_B = np.dot(R_B.T, P_world - t_B)
            
            # Apparent motion seen on the camera sensor:
            dx_val = -P_B[0]
            dy_val = -P_B[1]
        else:
            # Pure linear translation (baseline before working_distance feature)
            dx_val = delta_local_pure[0]
            dy_val = delta_local_pure[1]
            
        dz_val = delta_local_pure[2]
        selected_deltas.append([dx_val, dy_val, dz_val])
        
    selected_deltas = np.array(selected_deltas)
    pure_deltas = np.array(pure_deltas)
    
    if len(selected_deltas) > 0:
        dx = selected_deltas[:, 0]
        dy = selected_deltas[:, 1]
        dz = selected_deltas[:, 2]
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

