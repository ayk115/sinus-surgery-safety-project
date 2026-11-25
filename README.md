# Sinus Surgery Safety Project

Tools and notebooks for analyzing endoscopic camera motion in sinus surgery videos by aligning 2D Lucas-Kanade optical flow with 3D camera trajectory poses.

## Repository Structure

- `optical_flow_utils.py` — Endoscopic viewport circle masking (`extract_circle_mask`), frame extraction, Lucas-Kanade optical flow computation, and dynamic motion-based keyframe selection/aggregation.
- `trajectory_utils.py` — Camera pose (`4x4` matrix) loading, distance/index-based keyframe filtering, and local camera-frame translation delta computation.
- `common_utils.py` — Shared math helpers (e.g., shortest angular difference).
- `translation.ipynb` — Main analysis notebook comparing 2D optical flow vectors against 3D camera trajectory deltas across samples.
- `optical_flow.ipynb` — Standalone notebook for visualizing tracked Lucas-Kanade optical flow features inside the circular endoscopic mask.
- `requirements.txt` — Python dependencies.

## Setup

1. **Create and activate a virtual environment:**
   ```zsh
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```zsh
   pip install -r requirements.txt
   ```


