import os
import json
import numpy as np
import cv2
import vtk
import vtk.util.numpy_support as vtk_np
import slicer

print("="*60)
print("3D SLICER ENDOSCOPIC SIMULATION GENERATOR")
print("="*60)

# -------------------------------------------------------------
# 1. Paths Configuration
# -------------------------------------------------------------
base_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
stl_path = os.path.join(base_dir, "dataset/CT/left_preop_1_618467211/segmentations/Sinus_both.stl")
traj_path = os.path.join(base_dir, "dataset/sample_1/sample_1_traj.json")
output_video_path = os.path.join(base_dir, "dataset/sample_1/sample_1_slicer_sim.mp4")
output_frames_dir = os.path.join(base_dir, "dataset/sample_1/sample_1_slicer_sim_frames")

os.makedirs(output_frames_dir, exist_ok=True)

# -------------------------------------------------------------
# 2. Ensure Airway Model is Loaded & Styled
# -------------------------------------------------------------
try:
    airway_node = slicer.util.getNode("Sinus_both")
    print("Found existing 'Sinus_both' model in scene.")
except:
    print(f"Loading airway model from: {stl_path} ...")
    airway_node = slicer.util.loadModel(stl_path)

disp = airway_node.GetDisplayNode()
disp.SetColor(0.85, 0.45, 0.45)        # Mucosal soft reddish-pink
disp.SetAmbient(0.25)
disp.SetDiffuse(0.8)
disp.SetSpecular(0.35)
if hasattr(disp, 'SetPower'):
    disp.SetPower(30.0)                 # Specular highlight power
disp.SetBackfaceCulling(False)          # Show inner lumen walls

# -------------------------------------------------------------
# 3. Setup Layout to Full 3D View for Maximum Resolution
# -------------------------------------------------------------
layoutManager = slicer.app.layoutManager()
original_layout = layoutManager.layout
if hasattr(slicer.vtkMRMLLayoutNode, 'SlicerLayoutOneUp3DView'):
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
elif hasattr(slicer.vtkMRMLLayoutNode, 'SlicerLayoutOne3DView'):
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOne3DView)
slicer.app.processEvents()

threeDWidget = layoutManager.threeDWidget(0)
threeDView = threeDWidget.threeDView()
renderWindow = threeDView.renderWindow()
renderer = renderWindow.GetRenderers().GetFirstRenderer()

cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(threeDView.mrmlViewNode())
camera = cameraNode.GetCamera()

# Set rigid sinus endoscope optics
camera.SetUseHorizontalViewAngle(True)
camera.SetViewAngle(75.0)                # 75-degree wide angle
camera.SetClippingRange(0.1, 250.0)      # Near-clip 0.1mm avoids wall clipping

# Coaxial camera light (moves with endoscope tip)
renderer.RemoveAllLights()
light = vtk.vtkLight()
light.SetLightTypeToCameraLight()
light.SetColor(1.0, 0.96, 0.92)         # Warm surgical LED
light.SetIntensity(1.15)
renderer.AddLight(light)

# -------------------------------------------------------------
# 4. Load Trajectory Poses
# -------------------------------------------------------------
print(f"Loading trajectory: {traj_path} ...")
with open(traj_path, "r") as f:
    traj = json.load(f)

frame_keys = sorted(traj.keys())
total_frames = len(frame_keys)
print(f"Loaded {total_frames} camera poses from trajectory.")

# -------------------------------------------------------------
# 5. Initialize VTK Window Capture & Video Writer
# -------------------------------------------------------------
renderWindow.Render()
wti = vtk.vtkWindowToImageFilter()
wti.SetInput(renderWindow)
wti.SetInputBufferTypeToRGB()
wti.ReadFrontBufferOff()
wti.Update()

dims = wti.GetOutput().GetDimensions()
width, height = dims[0], dims[1]
print(f"Rendering Resolution: {width} x {height} at 10.0 FPS (matches real-time surgical capture)")

fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# -------------------------------------------------------------
# 6. Render & Export Trajectory Video
# -------------------------------------------------------------
print(f"Generating simulation video: {output_video_path}")
look_distance = 15.0  # mm ahead along optical axis

for idx, fkey in enumerate(frame_keys):
    T = np.array(traj[fkey], dtype=float)
    
    pos = T[:3, 3]
    forward = T[:3, 2]     # Z-axis is forward optical axis
    up = -T[:3, 1]         # -Y is up in camera frame
    
    camera.SetPosition(pos[0], pos[1], pos[2])
    camera.SetFocalPoint(
        pos[0] + forward[0] * look_distance,
        pos[1] + forward[1] * look_distance,
        pos[2] + forward[2] * look_distance
    )
    camera.SetViewUp(up[0], up[1], up[2])
    
    renderWindow.Render()
    slicer.app.processEvents()
    
    # Capture RGB buffer
    wti.Modified()
    wti.Update()
    
    vtk_img = wti.GetOutput()
    scalars = vtk_img.GetPointData().GetScalars()
    frame_rgb = vtk_np.vtk_to_numpy(scalars).reshape((height, width, 3))
    frame_rgb = np.flipud(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Write to MP4 video
    writer.write(frame_bgr)
    
    if (idx + 1) % 100 == 0 or idx == total_frames - 1:
        print(f"  Rendered frame {idx + 1}/{total_frames} ({(idx + 1)/total_frames*100:.1f}%)")

writer.release()
layoutManager.setLayout(original_layout)
slicer.app.processEvents()

print("="*60)
print(f"SUCCESS! Simulated video saved to:")
print(f"  {output_video_path}")
print(f"Total Frames: {total_frames} | Resolution: {width}x{height}")
print("="*60)
