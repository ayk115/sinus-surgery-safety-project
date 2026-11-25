import cv2
import numpy as np

def extract_circle_mask(ref_img, pos_circle=0):
    """
    Detects the circular endoscopic viewport in a reference frame using the
    Hough Circle Transform and creates a binary mask (shrunk by 15 pixels to
    exclude border artifacts) for restricting feature tracking.
    
    Parameters:
    - ref_img: Reference video frame as a NumPy array (BGR format)
    - pos_circle: Index of the detected circle to use (default: 0, highest-voted circle)
    
    Returns:
    - Binary mask (uint8 NumPy array with 255 inside the circle and 0 outside),
      or None if no circles are detected.
    """
    print("Obtaining circle mask...")
    max_rad = ref_img.shape[1]
    img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    detected_circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=60,
        param2=50,
        minRadius=1,
        maxRadius=max_rad,
    )
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        cx, cy, radius = (
            detected_circles[0, pos_circle, 0],
            detected_circles[0, pos_circle, 1],
            detected_circles[0, pos_circle, 2],
        )
        circle_mask = np.full((ref_img.shape[0], ref_img.shape[1]), 0, dtype=np.uint8)
        cv2.circle(circle_mask, (int(cx), int(cy)), int(radius) - 15, (255, 255, 255), -1)
        return circle_mask
    else:
        print("No circles detected. Choose different reference image.")
        return None

def extract_video_frames(video_path, return_fps=False):
    """
    Reads a video file and returns all frames as NumPy arrays (BGR format),
    optionally along with the video's frame rate.

    Parameters:
    - video_path: Path to the video file
    - return_fps: If True, also return the detected FPS (default: False)

    Returns:
    - frames: List of video frames as NumPy arrays (BGR format), or (frames, fps)
      when return_fps is True.
    """
    print(f"Reading video: {video_path}")
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 30.0
        
    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    
    if return_fps:
        return frames, fps
    return frames

def _default_tracking_params(feature_params=None, lk_params=None, quality_level=0.02):
    """
    Returns Shi-Tomasi feature-detection and Lucas-Kanade optical-flow parameter
    dicts, filling in defaults when not provided.
    """
    if feature_params is None:
        feature_params = {
            "maxCorners": 250,
            "qualityLevel": quality_level,
            "minDistance": 7,
            "blockSize": 7,
        }
    if lk_params is None:
        lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
    return feature_params, lk_params


def compute_optical_flow(video_frames, feature_params=None, lk_params=None):
    """
    Runs the Lucas-Kanade optical flow algorithm on a list of video frames,
    computing the robust median displacement of all tracked tissue features.
    
    Parameters:
    - video_frames: List of video frames as NumPy arrays (BGR format)
    - feature_params: Optional Shi-Tomasi feature-detection settings
    - lk_params: Optional Lucas-Kanade optical-flow settings
                  
    Returns:
    - NumPy array of shape (N-1, 2) containing [flow_x, flow_y] for each frame transition.
    """
    feature_params, lk_params = _default_tracking_params(
        feature_params, lk_params, quality_level=0.02
    )

    if not video_frames:
        return []

    print(f"Computing optical flow across {len(video_frames)} frames (Feature median)...")
    
    previous_frame = video_frames[0]
    circle_mask = extract_circle_mask(previous_frame)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    points_previous = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=circle_mask,
        **feature_params,
    )

    optical_flow_results = []
    
    for video_frame_index in range(1, len(video_frames)):
        current_frame = video_frames[video_frame_index]
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Re-detect features if the count drops below 50
        if points_previous is None or len(points_previous) < 50:
            points_previous = cv2.goodFeaturesToTrack(
                previous_gray,
                mask=circle_mask,
                **feature_params,
            )

        if points_previous is not None and len(points_previous) > 0:
            points_current, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points_previous,
                None,
                **lk_params,
            )

            if points_current is not None and status is not None:
                valid = status.ravel() == 1
                # Enforce a strict error margin to drop bad tracking
                if error is not None:
                    valid &= error.ravel() < 30

                good_previous = points_previous[valid].reshape(-1, 2)
                good_current = points_current[valid].reshape(-1, 2)
            else:
                good_previous = np.empty((0, 2), dtype=np.float32)
                good_current = np.empty((0, 2), dtype=np.float32)
        else:
            good_previous = np.empty((0, 2), dtype=np.float32)
            good_current = np.empty((0, 2), dtype=np.float32)

        if len(good_current) > 0:
            flow_vectors = good_current - good_previous
            flow_x = float(np.median(flow_vectors[:, 0]))
            flow_y = float(np.median(flow_vectors[:, 1]))
            optical_flow_results.append([flow_x, flow_y])
            points_previous = good_current.reshape(-1, 1, 2)
        else:
            optical_flow_results.append([np.nan, np.nan])
            points_previous = None

        previous_gray = current_gray

    return np.array(optical_flow_results)


def filter_keyframes_by_flow(
    optical_flow_results,
    pixel_threshold=50.0,
    angle_threshold_deg=None,
    min_turn_pixels=15.0,
    min_interval=4,
    max_interval=40,
):
    """
    Selects keyframe indices purely based on visual motion (optical flow) in the video.
    A new keyframe is pinned whenever:
    - Condition A: Cumulative 2D pixel displacement reaches or exceeds `pixel_threshold` pixels.
    - Condition B: Direction angle changes by >= `angle_threshold_deg` (gated by `min_turn_pixels`).
    - Timeout: `frames_elapsed` reaches `max_interval`.
    Guarded by:
    - Minimum duration: `frames_elapsed` >= `min_interval` to prevent single-frame glitch bursts.
    
    Parameters:
    - optical_flow_results: Array of (N-1, 2) frame-to-frame flow vectors [flow_x, flow_y]
    - pixel_threshold: Cumulative pixel displacement threshold (default: 50.0 px)
    - angle_threshold_deg: Maximum directional angle shift before triggering keyframe (default: None)
    - min_turn_pixels: Minimum pixel motion required to reliably evaluate angle change (default: 15.0 px)
    - min_interval: Minimum frames required between keyframes (default: 4 frames)
    - max_interval: Maximum frames allowed before forcing a keyframe (default: 40 frames)
    
    Returns:
    - keyframe_indices: List of integer frame indices for selected keyframes (starting at 0)
    """
    flow_vectors = np.nan_to_num(optical_flow_results, nan=0.0)
    keyframe_indices = [0]
    accum_x = 0.0
    accum_y = 0.0
    last_direction_angle = None
    angle_threshold_rad = np.radians(angle_threshold_deg) if angle_threshold_deg is not None else None
    
    for i in range(len(flow_vectors)):
        accum_x += flow_vectors[i, 0]
        accum_y += flow_vectors[i, 1]
        
        pixel_distance = np.sqrt(accum_x**2 + accum_y**2)
        frames_elapsed = (i + 1) - keyframe_indices[-1]
        
        # Calculate current running direction angle (in image coordinates: -y, -x matches camera motion)
        current_direction_angle = np.arctan2(-accum_y, -accum_x)
        
        # Condition A: Cumulative pixel distance threshold reached
        trigger_distance = pixel_distance >= pixel_threshold
        
        # Condition B: Direction angle changed significantly from previous interval (optional)
        trigger_angle = False
        if (
            angle_threshold_rad is not None
            and last_direction_angle is not None
            and pixel_distance >= min_turn_pixels
        ):
            diff = np.abs(np.arctan2(np.sin(current_direction_angle - last_direction_angle),
                                     np.cos(current_direction_angle - last_direction_angle)))
            if diff >= angle_threshold_rad:
                trigger_angle = True
                
        # Timeout condition
        trigger_timeout = frames_elapsed >= max_interval
        
        # Guard: Only allow triggering if minimum interval has passed
        should_trigger = trigger_distance or trigger_angle or trigger_timeout
        if (min_interval is None or frames_elapsed >= min_interval) and should_trigger:
            keyframe_indices.append(i + 1)
            last_direction_angle = current_direction_angle
            accum_x = 0.0
            accum_y = 0.0
            
    return keyframe_indices


def aggregate_optical_flow(optical_flow_results, interval=15, keyframe_indices=None):
    """
    Takes the frame-by-frame optical flow vectors and aggregates (sums) them 
    over a specified interval (e.g., 15 frames) or across dynamic keyframe indices
    to match a filtered trajectory.
    
    Parameters:
    - optical_flow_results: Array of frame-to-frame flow vectors
    - interval: Fixed frame interval (used if keyframe_indices is None, default: 15)
    - keyframe_indices: Optional list of frame numbers defining dynamic windows
    
    Returns a dictionary containing:
    - 'flow_x': Array of summed X vectors for each interval
    - 'flow_y': Array of summed Y vectors for each interval
    - 'magnitudes': Array of the total magnitude for each interval
    - 'angles': Array of the directional angle (in radians) for each interval
    """
    # Replace any NaNs with 0.0 to safely sum across intervals
    flow_vectors = np.nan_to_num(optical_flow_results, nan=0.0)
    
    chunked_x = []
    chunked_y = []
    
    # 1. Dynamic Keyframe Windows
    if keyframe_indices is not None and len(keyframe_indices) > 1:
        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]
            if start_idx >= len(flow_vectors):
                break
            chunk = flow_vectors[start_idx : min(end_idx, len(flow_vectors))]
            total_x = np.sum(chunk[:, 0]) if len(chunk) > 0 else 0.0
            total_y = np.sum(chunk[:, 1]) if len(chunk) > 0 else 0.0
            chunked_x.append(total_x)
            chunked_y.append(total_y)
            
    # 2. Fixed Interval (Fallback)
    elif interval is not None and interval > 0:
        num_intervals = len(flow_vectors) // interval
        for i in range(num_intervals):
            start_idx = i * interval
            end_idx = start_idx + interval
            chunk = flow_vectors[start_idx:end_idx]
            total_x = np.sum(chunk[:, 0])
            total_y = np.sum(chunk[:, 1])
            chunked_x.append(total_x)
            chunked_y.append(total_y)
            
    chunked_x = np.array(chunked_x)
    chunked_y = np.array(chunked_y)
    
    # Calculate overall magnitude and angle for the chunks
    magnitudes = np.sqrt(chunked_x**2 + chunked_y**2)
    angles = np.arctan2(-chunked_y, -chunked_x)
    
    return {
        "flow_x": chunked_x,
        "flow_y": chunked_y,
        "magnitudes": magnitudes,
        "angles": angles
    }


def render_optical_flow_video(
    input_video,
    output_video,
    feature_params=None,
    lk_params=None,
    min_features=20,
):
    """
    Tracks sparse tissue features across an endoscopic video using Lucas-Kanade
    optical flow inside the circular viewport mask and writes an annotated output
    video showing tracked points and their motion trails.

    Parameters:
    - input_video: Path to the source video file
    - output_video: Path where the annotated .mp4 video will be saved
    - feature_params: Optional Shi-Tomasi feature-detection settings
    - lk_params: Optional Lucas-Kanade optical-flow settings
    - min_features: Minimum tracked feature count before re-detecting corners (default: 20)

    Returns:
    - Dictionary containing 'output_video', 'frames_processed', 'width', 'height',
      'fps', and 'tracked_point_total'.
    """
    feature_params, lk_params = _default_tracking_params(
        feature_params, lk_params, quality_level=0.2
    )

    video_frames, fps = extract_video_frames(input_video, return_fps=True)
    if not video_frames:
        raise RuntimeError(f"Could not read any frames from video: {input_video}")

    previous_frame = video_frames[0]
    height, width = previous_frame.shape[:2]
    total_frames = len(video_frames)
    duration_ms = (total_frames / fps) * 1000

    print(f"Input video opened: {width} x {height} at {fps:.2f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Video duration: {duration_ms:.2f} ms")

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    try:
        circle_mask = extract_circle_mask(previous_frame)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        points_previous = cv2.goodFeaturesToTrack(
            previous_gray,
            mask=circle_mask,
            **feature_params,
        )

        trail_layer = np.zeros_like(previous_frame)
        colors = np.random.default_rng(7).integers(
            0,
            256,
            size=(feature_params["maxCorners"], 3),
        )

        frame_number = 1
        tracked_point_total = 0
        print(
            f"Initial features detected: "
            f"{0 if points_previous is None else len(points_previous)}"
        )

        for current_frame in video_frames[1:]:
            current_frame = current_frame.copy()
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            if points_previous is not None and len(points_previous) > 0:
                points_current, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    current_gray,
                    points_previous,
                    None,
                    **lk_params,
                )

                if points_current is not None and status is not None:
                    valid = status.ravel() == 1
                    good_current = points_current[valid]
                    good_previous = points_previous[valid]
                    tracked_point_total += len(good_current)

                    for point_index, (current, previous) in enumerate(
                        zip(good_current, good_previous)
                    ):
                        x_current, y_current = current.ravel().astype(int)
                        x_previous, y_previous = previous.ravel().astype(int)
                        color = tuple(
                            int(value) for value in colors[point_index % len(colors)]
                        )
                        cv2.line(
                            trail_layer,
                            (x_previous, y_previous),
                            (x_current, y_current),
                            color,
                            2,
                        )
                        cv2.circle(
                            current_frame,
                            (x_current, y_current),
                            4,
                            color,
                            -1,
                        )

                    points_previous = (
                        good_current.reshape(-1, 1, 2)
                        if len(good_current) > 0
                        else None
                    )
                else:
                    points_previous = None

            if points_previous is None or len(points_previous) < min_features:
                points_previous = cv2.goodFeaturesToTrack(
                    current_gray,
                    mask=circle_mask,
                    **feature_params,
                )

            writer.write(cv2.add(current_frame, trail_layer))
            previous_gray = current_gray
            frame_number += 1
    finally:
        writer.release()

    print(f"Processed {frame_number} frames")
    print(f"Frame size: {width} x {height}")
    print(f"Accumulated tracked points: {tracked_point_total}")
    print(f"Output video saved as: {output_video}")

    return {
        "output_video": output_video,
        "frames_processed": frame_number,
        "width": width,
        "height": height,
        "fps": fps,
        "tracked_point_total": tracked_point_total,
    }
