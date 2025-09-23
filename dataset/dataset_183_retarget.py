import nimblephysics as nimble
import numpy as np
import os
import random
import json
import torch
from glob import glob
from torch.utils import data
from tqdm import tqdm
import functools

class Retargeted183Dataset(data.Dataset):
    def __init__(self, window_size=64, unit_length=4, data_dir='/home/mnt/datasets/183_retargeted', pre_load=False):
        self.window_size = window_size
        self.unit_length = unit_length
        self.data_dir = data_dir
        self.pre_load = pre_load

        # defaults for biomechanical vectorization
        self.default_biomech_target_samples = 20   # temporal samples per channel
        self.default_biomech_dim = 2048            # final fixed vector length (pad/truncate)

        self.b3d_file_paths = glob(os.path.join(data_dir, '*.b3d'))
        print(f"Found {len(self.b3d_file_paths)} b3d files in {data_dir}")

        self.subject_metadata = {}
        self.subject_biomech = {}  # If pre_load: biomech dict; else: {subject_id: b3d_file}
        self.subject_skeleton_raw = {}
        self.subject_skeleton = {}
        self.motion_mapping = []
        self._preloaded_motions = [] if pre_load else None

        self._preload_data()
        print(f"Total number of motions: {len(self.motion_mapping)}")

    def _preload_data(self):
        for b3d_file in tqdm(self.b3d_file_paths, desc="Preprocessing data"):
            try:
                if os.path.getsize(b3d_file) == 0:
                    print(f"Skipping empty file: {b3d_file}")
                    continue
                subject_id = os.path.splitext(os.path.basename(b3d_file))[0]
                subject = nimble.biomechanics.SubjectOnDisk(b3d_file)
                num_trials = subject.getNumTrials()
                notes = subject.getNotes()
                sport = "unknown"
                level = "unknown"
                injuries = []
                if notes:
                    try:
                        notes_dict = json.loads(notes)
                        sport = notes_dict.get("sport", "unknown")
                        level = notes_dict.get("level", "unknown")
                        injuries = notes_dict.get("injuries", [])
                    except Exception as e:
                        print(f"Failed to parse notes for {subject_id}: {e}")

                if subject_id not in self.subject_metadata:
                    self.subject_metadata[subject_id] = {
                        "age": subject.getAgeYears(),
                        "height": subject.getHeightM(),
                        "mass": subject.getMassKg(),
                        "sex": subject.getBiologicalSex(),
                        "sport": sport,
                        "level": level,
                        "injuries": injuries
                    }
                # Biomech: only pre-load if requested
                if self.pre_load:
                    if subject_id not in self.subject_biomech:
                        self._extract_biomech_features(subject, subject_id, num_trials)
                else:
                    # Store file path for lazy loading
                    if subject_id not in self.subject_biomech:
                        self.subject_biomech[subject_id] = b3d_file

                if subject_id not in self.subject_skeleton_raw:
                    self._extract_skeleton_info(subject, subject_id)

                for trial_idx in range(num_trials):
                    trial_length = subject.getTrialLength(trial_idx)
                    if trial_length < self.window_size:
                        continue

                    seconds_per_frame = subject.getTrialTimestep(trial_idx)
                    fps = int(round(1.0 / seconds_per_frame)) if seconds_per_frame > 0 else 0

                    self.motion_mapping.append({
                        'file_path': b3d_file,
                        'trial_idx': trial_idx,
                        'trial_length': trial_length,
                        'fps': fps,
                        'subject_id': subject_id,
                        'name': subject.getTrialName(trial_idx)
                    })

                    if self.pre_load:
                        try:
                            frames = subject.readFrames(
                                trial=trial_idx,
                                startFrame=0,
                                numFramesToRead=trial_length,
                                includeSensorData=False,
                                includeProcessingPasses=True
                            )
                            kin_passes = [frame.processingPasses[0] for frame in frames if frame.processingPasses]
                            positions = np.array([kp.pos for kp in kin_passes if hasattr(kp, 'pos')])
                            target_fps = 120
                            if fps > target_fps:
                                step = int(round(fps / target_fps))
                                positions = positions[::step]
                            self._preloaded_motions.append(positions)
                        except Exception as e:
                            print(f"Failed to load motion for {b3d_file} trial {trial_idx}: {e}")
                            self._preloaded_motions.append(None)
            except Exception as e:
                print(f"Skipping file {b3d_file} due to error: {e}")

        if self.subject_skeleton_raw:
            vals = np.stack(list(self.subject_skeleton_raw.values()))
            mean = vals.mean(axis=0)
            std = vals.std(axis=0, ddof=0)
            std_safe = np.where(std == 0, 1.0, std)
            self.subject_skeleton = {k: (v - mean) / std_safe for k, v in self.subject_skeleton_raw.items()}

        print(f"Total number of motions: {len(self.motion_mapping)}")

    def _extract_biomech_features(self, subject, subject_id, num_trials):
        velocities = []
        grf_mags = []
        com_vel = []
        com_acc = []

        for trial_idx in range(num_trials):
            trial_length = subject.getTrialLength(trial_idx)
            frames = subject.readFrames(
                trial=trial_idx,
                startFrame=0,
                numFramesToRead=trial_length,
                includeSensorData=False,
                includeProcessingPasses=True
            )
            for frame in frames:
                if not frame.processingPasses:
                    continue

                v = np.abs(frame.processingPasses[0].vel)
                if not (np.any(np.isnan(v)) or np.any(np.isinf(v))):
                    velocities.append(v)

                grf = getattr(frame.processingPasses[0], "groundContactForce", None)
                if grf is not None and len(grf) > 0:
                    grf = np.array(grf).reshape(-1, 3)
                    grf_norm = np.linalg.norm(grf, axis=1)
                    grf_mags.append(grf_norm)

                cv = getattr(frame.processingPasses[0], "comVel", None)
                if cv is not None:
                    cv_arr = np.array(cv)
                    if not (np.any(np.isnan(cv_arr)) or np.any(np.isinf(cv_arr))):
                        com_vel.append(cv_arr)

                ca = getattr(frame.processingPasses[0], "comAcc", None)
                if ca is not None:
                    ca_arr = np.array(ca)
                    if not (np.any(np.isnan(ca_arr)) or np.any(np.isinf(ca_arr))):
                        com_acc.append(ca_arr)

        biomech = {
            "joint_velocities": velocities,
            "ground_contact_forces": grf_mags,
            "com_vel": com_vel,
            "com_acc": com_acc
        }

        # Build a fixed-length biomechanical vector via temporal downsampling and store alongside raw data.
        try:
            biomech_vector = self._biomech_to_vector(biomech,
                                                     target_samples=self.default_biomech_target_samples,
                                                     biomech_dim=self.default_biomech_dim)
            biomech['vector'] = biomech_vector
        except Exception:
            biomech['vector'] = np.zeros(self.default_biomech_dim, dtype=np.float32)

        self.subject_biomech[subject_id] = biomech

    def _get_biomech_lazy(self, subject_id):
        # Only used if pre_load=False
        b3d_file = self.subject_biomech[subject_id]
        subject = nimble.biomechanics.SubjectOnDisk(b3d_file)
        num_trials = subject.getNumTrials()
        self._extract_biomech_features(subject, subject_id, num_trials)
        return self.subject_biomech[subject_id]

    def _extract_skeleton_info(self, subject, subject_id):
        """Extract skeleton information for a subject"""
        bodydict = {}
        skeleton = subject.readSkel(0, ignoreGeometry=True)

        for i in range(skeleton.getNumBodyNodes()):
            node = skeleton.getBodyNode(i)
            name = node.getName()
            pos = node.getWorldTransform().translation()
            bodydict[name] = pos

        def safe_length(part1, part2):
            if part1 in bodydict and part2 in bodydict:
                return float(np.linalg.norm(bodydict[part1] - bodydict[part2]))
            else:
                return 0.0

        # Lower limbs
        r_hip = safe_length("pelvis", "femur_r")    # right hip
        r_thigh = safe_length("femur_r", "tibia_r")    # right thigh
        r_shank = safe_length("tibia_r", "talus_r")    # right shank
        r_foot_rear = safe_length("talus_r", "calcn_r")    # right foot rear
        r_forefoot = safe_length("calcn_r", "toes_r")     # right forefoot

        l_hip = safe_length("pelvis", "femur_l")     # left hip
        l_thigh = safe_length("femur_l", "tibia_l")    # left thigh
        l_shank = safe_length("tibia_l", "talus_l")    # left shank
        l_foot_rear = safe_length("talus_l", "calcn_l")    # left foot rear
        l_forefoot = safe_length("calcn_l", "toes_l")     # left forefoot

        # Upper limbs
        r_shoulder = safe_length("torso", "humerus_r")    # right shoulder
        r_upper_arm = safe_length("humerus_r", "ulna_r")   # right upper arm
        r_forearm = safe_length("ulna_r", "radius_r")    # right forearm
        r_hand = safe_length("radius_r", "hand_r")    # right hand

        l_shoulder = safe_length("torso", "humerus_l")    # left shoulder
        l_upper_arm = safe_length("humerus_l", "ulna_l")   # left upper arm
        l_forearm = safe_length("ulna_l", "radius_l")    # left forearm
        l_hand = safe_length("radius_l", "hand_l")    # left hand

        torso_length = safe_length("pelvis", "torso")    # torso

        skeleton_lengths = [
            r_hip, r_thigh, r_shank, r_foot_rear, r_forefoot,
            l_hip, l_thigh, l_shank, l_foot_rear, l_forefoot,
            r_shoulder, r_upper_arm, r_forearm, r_hand,
            l_shoulder, l_upper_arm, l_forearm, l_hand,
            torso_length
        ]

        self.subject_skeleton_raw[subject_id] = skeleton_lengths

    def _load_motion_data(self, motion_info):
        # Always load the entire trial at once
        subject = nimble.biomechanics.SubjectOnDisk(motion_info['file_path'])
        trial_idx = motion_info['trial_idx']
        trial_length = motion_info['trial_length']

        frames = subject.readFrames(
            trial=trial_idx,
            startFrame=0,
            numFramesToRead=trial_length,
            includeSensorData=False,
            includeProcessingPasses=True
        )
        kin_passes = [frame.processingPasses[0] for frame in frames if frame.processingPasses]
        positions = np.array([kp.pos for kp in kin_passes if hasattr(kp, 'pos')])

        # Resample if needed (120hz)
        fps = motion_info['fps']
        target_fps = 120
        if fps > target_fps:
            step = int(round(fps / target_fps))
            positions = positions[::step]

        return positions

    def __len__(self):
        return len(self.motion_mapping)

    def __getitem__(self, item):
        motion_info = self.motion_mapping[item]

        # Use pre-loaded motions if available
        if self.pre_load and self._preloaded_motions is not None:
            motion = self._preloaded_motions[item]
            if motion is None:
                raise ValueError(f"Pre-loaded motion {item} is None ({motion_info['name']})")
        else:
            motion = self._load_motion_data(motion_info)

        # Get metadata
        subject_id = motion_info['subject_id']
        name = motion_info['name']

        len_motion = len(motion) if len(motion) <= self.window_size else self.window_size

        # Crop or pad to window_size
        if len(motion) >= self.window_size:
            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx:idx + self.window_size]
        else:
            repeat_count = (self.window_size + len(motion) - 1) // len(motion)
            motion = np.tile(motion, (repeat_count, 1))[:self.window_size]

        if np.any(np.isnan(motion)) or np.any(np.isinf(motion)):
            raise ValueError(f"NaN or Inf detected in motion sample {item} ({motion_info['name']})")

        motion = torch.from_numpy(motion).float()
        return motion, len_motion, name, subject_id

    def get_biomech(self, subject_id):
        """
        Get biomechanics for a subject. Loads on demand if not preloaded.
        """
        if self.pre_load:
            return self.subject_biomech.get(subject_id, {})
        else:
            if isinstance(self.subject_biomech.get(subject_id), dict):
                return self.subject_biomech[subject_id]
            else:
                return self._get_biomech_lazy(subject_id)

    def _temporal_downsample(self, arr_list, target_samples, expected_channels=None, method='uniform'):
        """
        Convert list-of-frame arrays -> (target_samples, C) numpy array.
        - arr_list: list of per-frame arrays (each 1D or scalar)
        - expected_channels: if provided, pad/truncate channels to this width
        - method: 'uniform' picks uniform indices, 'mean' pools equal-sized chunks
        """
        if len(arr_list) == 0:
            C = expected_channels or 1
            return np.zeros((target_samples, C), dtype=np.float32)
        arr = np.array(arr_list, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        N, C = arr.shape
        if expected_channels is not None and C != expected_channels:
            if C < expected_channels:
                pad = np.zeros((N, expected_channels - C), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
                C = expected_channels
            else:
                arr = arr[:, :expected_channels]
                C = expected_channels
        if N == target_samples:
            return arr
        if method == 'uniform' or N > target_samples:
            idx = np.linspace(0, N - 1, target_samples).astype(int)
            return arr[idx]
        # method == 'mean' and N < target_samples -> upsample by repeating last row to match
        repeats = target_samples - N
        pad = np.tile(arr[-1:], (repeats, 1))
        return np.concatenate([arr, pad], axis=0)

    def _biomech_to_vector(self, biomech_dict, target_samples=20, include_stats=True, biomech_dim=None):
        """
        Create a deterministic 1D biomech vector:
        - downsample joint_vel, com_vel, com_acc to target_samples each (preserving channels)
        - reduce ground contact forces per-frame to a scalar (sum) then downsample
        - concatenate flattened samples, optionally append simple stats, pad/truncate to biomech_dim
        """
        parts = []

        def infer_channels(lst, fallback=1):
            for x in lst:
                a = np.array(x)
                if a.ndim == 1:
                    return a.shape[0]
                elif a.ndim > 1:
                    return a.shape[1]
            return fallback

        # joint velocities
        jv_ch = infer_channels(biomech_dict.get("joint_velocities", []), fallback=1)
        jv_sampled = self._temporal_downsample(biomech_dict.get("joint_velocities", []),
                                               target_samples=target_samples,
                                               expected_channels=jv_ch,
                                               method='uniform')
        parts.append(jv_sampled.flatten())

        # com velocities and accelerations (expect 3 channels)
        cv_sampled = self._temporal_downsample(biomech_dict.get("com_vel", []),
                                               target_samples=target_samples,
                                               expected_channels=3, method='uniform')
        ca_sampled = self._temporal_downsample(biomech_dict.get("com_acc", []),
                                               target_samples=target_samples,
                                               expected_channels=3, method='uniform')
        parts.append(cv_sampled.flatten())
        parts.append(ca_sampled.flatten())

        # ground contact forces: reduce each frame to a scalar (sum of magnitudes) then sample
        grf_scalars = []
        for g in biomech_dict.get("ground_contact_forces", []):
            g = np.array(g)
            if g.size == 0:
                grf_scalars.append(0.0)
            else:
                grf_scalars.append(float(np.sum(g)))
        grf_sampled = self._temporal_downsample(grf_scalars, target_samples=target_samples, expected_channels=1, method='uniform')
        parts.append(grf_sampled.flatten())

        feat = np.concatenate(parts).astype(np.float32)
        if include_stats:
            feat = np.concatenate([feat,
                                   np.array([np.nanmean(feat), np.nanstd(feat), np.nanmax(feat), np.nanmin(feat)], dtype=np.float32)])
        if biomech_dim is not None:
            if feat.shape[0] < biomech_dim:
                feat = np.pad(feat, (0, biomech_dim - feat.shape[0]), mode='constant')
            else:
                feat = feat[:biomech_dim]
        return feat

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def retargeted183_data_loader(window_size=64, unit_length=4, batch_size=1, num_workers=0, data_dir='/home/mnt/datasets/183_retargeted', pre_load=False):
    """
    Create a DataLoader for the 183_retargeted dataset optimized for multiple workers.

    Args:
        window_size: Number of frames to include in each sample
        unit_length: Unit time length for processing
        batch_size: Batch size for the data loader
        num_workers: Number of worker threads for loading data
        data_dir: Directory containing the b3d files
        pre_load: If True, load all motions into memory at initialization

    Returns:
        DataLoader object for the dataset
    """
    dataset = Retargeted183Dataset(window_size=window_size, unit_length=unit_length, data_dir=data_dir, pre_load=pre_load)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False if num_workers == 0 else True,
        pin_memory=True,
        drop_last=True
    )
    return loader

def main():
    # Example usage with multiple workers
    dataloader = retargeted183_data_loader(
        window_size=64,
        unit_length=4,
        batch_size=16,
        num_workers=4
    )

    # Test by loading a batch
    for batch in dataloader:
        motion, len_motion, name, subject_id = batch  # biomech removed
        print(f"Motion shape: {motion.shape}")
        print(f"Motion length: {len_motion}")
        print(f"Subject IDs: {subject_id}")

if __name__ == "__main__":
    main()