import cv2
import numpy as np

def extract_circle_mask(ref_img, pos_circle=0):
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

def extract_video_frames(video_path):
    """
    Reads a video file and returns a list of all frames as NumPy arrays (BGR format).
    """
    print(f"Reading video: {video_path}")
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
        
    capture.release()
    
    return frames

def compute_optical_flow(video_frames, feature_params=None, lk_params=None):
    """
    Runs the Lucas-Kanade optical flow algorithm on a list of video frames.
    
    Returns a NumPy array of shape (N, 2) containing the median 
    optical flow vectors [flow_x, flow_y] between consecutive frames.
    """
    if feature_params is None:
        feature_params = {
            "maxCorners": 250,
            "qualityLevel": 0.2,
            "minDistance": 7,
            "blockSize": 7,
        }
    if lk_params is None:
        lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    if not video_frames:
        return []

    print(f"Computing optical flow across {len(video_frames)} frames...")
    
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

        # IMPROVEMENT 1: Re-detect features if the count drops below 50!
        # If we rely on the count hitting exactly 0, we risk calculating the 
        # median flow of only 2 or 3 surviving pixels, which creates massive noise spikes.
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
                # IMPROVEMENT 2: Enforce a strict error margin to drop bad tracking
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

        if len(good_current) >= 4:
            # Estimate 2D rigid motion using OpenCV's built-in RANSAC
            transform, inliers = cv2.estimateAffinePartial2D(
                good_previous,
                good_current,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=500,
            )

            if transform is not None and inliers is not None and np.sum(inliers) >= 4:
                inlier_mask = inliers.ravel() == 1
                inlier_flows = good_current[inlier_mask] - good_previous[inlier_mask]
                
                # Compute the mean of the consensus inliers (averaging out sub-pixel noise)
                flow_x = float(np.mean(inlier_flows[:, 0]))
                flow_y = float(np.mean(inlier_flows[:, 1]))
                
                # Keep verified inliers for the next frame
                points_previous = good_current[inlier_mask].reshape(-1, 1, 2)
            else:
                # Fallback to median if RANSAC cannot find a consensus
                flow_vectors = good_current - good_previous
                flow_x = float(np.median(flow_vectors[:, 0]))
                flow_y = float(np.median(flow_vectors[:, 1]))
                points_previous = good_current.reshape(-1, 1, 2)

            optical_flow_results.append([flow_x, flow_y])
        elif len(good_current) > 0:
            flow_vectors = good_current - good_previous
            optical_flow_results.append([
                float(np.median(flow_vectors[:, 0])),
                float(np.median(flow_vectors[:, 1]))
            ])
            points_previous = good_current.reshape(-1, 1, 2)
        else:
            optical_flow_results.append([np.nan, np.nan])
            points_previous = None

        previous_gray = current_gray

    return np.array(optical_flow_results)

def aggregate_optical_flow(optical_flow_results, interval=15):
    """
    Takes the frame-by-frame optical flow vectors and aggregates (sums) them 
    over a specified interval (e.g., 15 frames) to match a filtered trajectory.
    
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
    
    # Calculate how many full chunks of 15 we can make
    num_intervals = len(flow_vectors) // interval
    
    for i in range(num_intervals):
        start_idx = i * interval
        end_idx = start_idx + interval
        
        chunk = flow_vectors[start_idx:end_idx]
        
        # Sum up the tiny frame-by-frame movements in this chunk
        total_x = np.sum(chunk[:, 0])
        total_y = np.sum(chunk[:, 1])
        
        chunked_x.append(total_x)
        chunked_y.append(total_y)
        
    chunked_x = np.array(chunked_x)
    chunked_y = np.array(chunked_y)
    
    # Calculate overall magnitude and angle for the 15-frame chunks
    magnitudes = np.sqrt(chunked_x**2 + chunked_y**2)
    angles = np.arctan2(-chunked_y, -chunked_x)
    
    return {
        "flow_x": chunked_x,
        "flow_y": chunked_y,
        "magnitudes": magnitudes,
        "angles": angles
    }
