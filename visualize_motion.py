import nimblephysics as nimble
import numpy as np
import time

# Load a subject's motion from a .b3d file using nimblephysics
#b3d_path = "/home/mnt/datasets/AddBiomechanics/train/No_Arm/Carter2023_Formatted_No_Arm/P003_split3/P003_split3.b3d"  # <-- Update this path
b3d_path = "/home/mnt/datasets/183_retargeted/1195.b3d"  # <-- Update this path

# Use SubjectOnDisk to read the first trial's poses
subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
trial_idx = 1
trial_length = subject.getTrialLength(trial_idx)
frames = subject.readFrames(
    trial=trial_idx,
    startFrame=0,
    numFramesToRead=trial_length,
    includeSensorData=False,
    includeProcessingPasses=True
)
kin_passes = [frame.processingPasses[0] for frame in frames if frame.processingPasses]
poses = np.array([kp.pos for kp in kin_passes if hasattr(kp, 'pos')])  # shape: (frames, dofs)

# Pad with zeros to match skeleton DOFs (total 37)
zeros = np.zeros((poses.shape[0], 37 - poses.shape[1]))
expanded = np.hstack([poses, zeros])    # Result: shape (frames, 37)

# # Load generated poses from .npy file
# poses = np.load("generated_motion.npy")  # various shapes possible
# print("raw poses shape:", poses.shape)


# def frames_from_poses(arr, target_dofs=37):
#     arr = np.asarray(arr)

#     # 1D vector -> single frame
#     if arr.ndim == 1:
#         if arr.size == target_dofs:
#             return arr[np.newaxis, :]
#         # pad or truncate
#         if arr.size < target_dofs:
#             pad = np.zeros(target_dofs - arr.size)
#             return np.hstack([arr, pad])[np.newaxis, :]
#         return arr[:target_dofs][np.newaxis, :]

#     # 2D array: could be (T, D) or (D, T)
#     if arr.ndim == 2:
#         if arr.shape[1] == target_dofs:
#             return arr.copy()
#         if arr.shape[0] == target_dofs:
#             return arr.T.copy()
#         # otherwise assume rows are frames and pad/truncate columns
#         if arr.shape[1] < target_dofs:
#             pad = np.zeros((arr.shape[0], target_dofs - arr.shape[1]))
#             return np.hstack([arr, pad])
#         return arr[:, :target_dofs]

#     # 3D arrays: common formats include (1, C, T), (B, T, C), (B, C, T)
#     if arr.ndim == 3:
#         # If batch dim 1, squeeze
#         if arr.shape[0] == 1:
#             arr = arr.squeeze(0)

#         # Now arr is 2D or 3D depending on original layout
#         if arr.ndim == 2:
#             # (C, T) or (T, C)
#             if arr.shape[0] == target_dofs:
#                 return arr.T.copy()
#             if arr.shape[1] == target_dofs:
#                 return arr.copy()
#             # fallback
#             if arr.shape[1] < target_dofs:
#                 pad = np.zeros((arr.shape[0], target_dofs - arr.shape[1]))
#                 return np.hstack([arr, pad])
#             return arr[:, :target_dofs]

#         # If still 3D (batch >1), try to move channel axis to last and flatten batch/time
#         # Find a dimension equal to target_dofs
#         axes = arr.shape
#         if target_dofs in axes:
#             chan_axis = int(np.where(np.array(axes) == target_dofs)[0][0])
#             moved = np.moveaxis(arr, chan_axis, -1)  # (..., T?, target_dofs)
#             frames = moved.reshape(-1, target_dofs)
#             return frames

#         # Fallback: collapse leading dims into time and use last axis as dofs
#         frames = arr.reshape(-1, arr.shape[-1])
#         if frames.shape[1] < target_dofs:
#             pad = np.zeros((frames.shape[0], target_dofs - frames.shape[1]))
#             return np.hstack([frames, pad])
#         return frames[:, :target_dofs]

#     raise ValueError(f'Unsupported poses ndim: {arr.ndim}')


# expanded = frames_from_poses(poses, target_dofs=37)
# print('Frames after conversion:', expanded.shape)

#Set up Nimble GUI and skeleton
gui = nimble.NimbleGUI()
gui.serve(8000)
rajagopal_opensim = nimble.RajagopalHumanBodyModel()
skeleton = rajagopal_opensim.skeleton
gui.nativeAPI().renderSkeleton(skeleton)

timestep = 1.0 / 240.0
# timestep = subject.getTrialTimestep(trial_idx) 

# Animate the motion in a loop forever
try:
    while True:
        for t in range(len(expanded)):
            skeleton.setPositions(expanded[t])
            gui.nativeAPI().renderSkeleton(skeleton)
            time.sleep(timestep)
finally:
    gui.blockWhileServing()