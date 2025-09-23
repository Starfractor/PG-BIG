import argparse
import json
import nimblephysics as nimble
import numpy as np
import os
import logging
from dataset.dataset_183_athletes import Subject183Dataset
from retarget_motions.optimize_marker_offsets import get_seed_offsets
from scipy.signal import butter, filtfilt
from multiprocessing import Pool
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_subject_metadata(subject):
    sport = getattr(subject, "sport", "unknown")
    level = getattr(subject, "level", "unknown")
    injuries = getattr(subject, "injuries", [])
    notes_dict = {
        "sport": sport,
        "level": level,
        "injuries": injuries,
    }
    return json.dumps(notes_dict)


def compute_offsets(subject, node_mapping, base_skeleton):
    """Compute per-body local marker offsets from the subject calibration using the Rajagopal fit
    approach. Compute offsets for each frame, then average (excluding top/bottom 5 percentile)."""
    offsets_map = {}

    # compute_offsets -> try to use calibration stored on subject
    calib = getattr(subject, "calibration", None)
    if calib is None or getattr(calib, "markers", None) is None:
        logging.warning(f"Subject {getattr(subject, 'ppid', 'unknown')} has no calibration; using seed offsets")
        valid_nodes = [node for node, marker in node_mapping.items() if marker is not None]
        seed_offsets = get_seed_offsets(valid_nodes)
        for i, node in enumerate(valid_nodes):
            offsets_map[node] = seed_offsets[i]
        for node in node_mapping:
            if node not in offsets_map:
                offsets_map[node] = np.zeros(3, dtype=np.float64)
        return offsets_map

    # collect all calibration frames
    calib_frames = calib.markers

    if len(calib_frames) == 0:
        logging.warning(f"No calibration frames for subject {getattr(subject, 'ppid', 'unknown')}; using seed offsets")
        valid_nodes = [node for node, marker in node_mapping.items() if marker is not None]
        seed_offsets = get_seed_offsets(valid_nodes)
        for i, node in enumerate(valid_nodes):
            offsets_map[node] = seed_offsets[i]
        for node in node_mapping:
            if node not in offsets_map:
                offsets_map[node] = np.zeros(3, dtype=np.float64)
        return offsets_map

    # Store offsets for each frame and each body node
    frame_offsets = {}  # {body_name: [offset_vectors]}
    
    for frame_idx, frame_markers in enumerate(calib_frames):
        # Build marker list for this frame
        markers_for_fit = []
        target_positions_list = []
        weights = []
        processed_node_names = []

        for body_name, marker_label in node_mapping.items():
            if marker_label is None:
                continue
            if marker_label not in frame_markers:
                continue
            
            coord = frame_markers[marker_label]
            if coord is None:
                continue
            coord_arr = np.asarray(coord, dtype=float)
            if coord_arr.size < 3:
                continue
            if np.all(np.isnan(coord_arr)):
                continue
            if np.allclose(coord_arr, 0.0):
                continue

            # Try body lookup APIs (robust)
            body = None
            try:
                body = base_skeleton.getBodyNode(body_name)
            except Exception:
                try:
                    body = base_skeleton.getBody(body_name)
                except Exception:
                    body = None
            if body is None:
                logging.debug(f"compute_offsets: body '{body_name}' not found on skeleton; skipping")
                continue

            offset_vec = np.zeros(3, dtype=np.float64)
            markers_for_fit.append((body, offset_vec))

            # Convert marker position to meters and apply frame transformation
            if np.linalg.norm(coord_arr) > 10.0:
                coord_m = coord_arr / 1000.0
            else:
                coord_m = coord_arr
            
            tp = np.array([coord_m[0], coord_m[1], coord_m[2]], dtype=np.float64).reshape(3,)
            target_positions_list.append(tp)
            weights.append(1.0)
            processed_node_names.append((body_name, marker_label))

        if len(target_positions_list) == 0:
            continue  # Skip this frame if no valid markers

        target_positions_mat = np.vstack(target_positions_list).astype(np.float64)
        target_positions_flat = target_positions_mat.flatten()
        marker_weights_np = np.array(weights, dtype=np.float64).reshape(-1, 1)

        # fit skeleton to this frame's marker positions
        try:
            base_skeleton.fitMarkersToWorldPositions(markers_for_fit, target_positions_flat, marker_weights_np, scaleBodies=True)
        except Exception as e:
            logging.debug(f"Frame {frame_idx} fit failed: {e}; skipping frame")
            continue

        # compute local offsets for this frame using fitted body transforms
        for (body_name, marker_label), (body, _) in zip(processed_node_names, markers_for_fit):
            world_pos = target_positions_list[processed_node_names.index((body_name, marker_label))]
            world_tf = body.getWorldTransform()

            # Try to extract R and t robustly
            R = None
            t = None
            try:
                if hasattr(world_tf, "rotation") and hasattr(world_tf, "translation"):
                    R = np.asarray(world_tf.rotation(), dtype=float)
                    t = np.asarray(world_tf.translation(), dtype=float).reshape(3,)
            except Exception:
                R = None
                t = None

            if R is None or t is None:
                try:
                    arr = np.asarray(world_tf)
                    if arr.shape == (4, 4):
                        R = arr[:3, :3].astype(float)
                        t = arr[:3, 3].astype(float).reshape(3,)
                except Exception:
                    R = None
                    t = None

            if R is None or t is None:
                continue  # Skip this marker for this frame

            local_offset_np = R.T.dot(world_pos - t)
            
            # Store this frame's offset for this body
            if body_name not in frame_offsets:
                frame_offsets[body_name] = []
            frame_offsets[body_name].append(local_offset_np.copy())

    # Now compute averaged offsets excluding top/bottom 5 percentile
    for body_name in node_mapping.keys():
        if body_name not in frame_offsets or len(frame_offsets[body_name]) == 0:
            # Fall back to seed offset for this body
            if node_mapping[body_name] is not None:
                valid_nodes = [body_name]
                seed_offsets = get_seed_offsets(valid_nodes)
                offsets_map[body_name] = seed_offsets[0]
            else:
                offsets_map[body_name] = np.zeros(3, dtype=np.float64)
            continue

        offset_vectors = np.array(frame_offsets[body_name])  # Shape: (n_frames, 3)
        
        if offset_vectors.shape[0] < 10:
            # If we have fewer than 10 frames, just use the mean
            avg_offset = np.mean(offset_vectors, axis=0)
        else:
            # Remove top and bottom 5 percentile for each coordinate
            filtered_offsets = []
            for coord_idx in range(3):
                coord_values = offset_vectors[:, coord_idx]
                p5 = np.percentile(coord_values, 5)
                p95 = np.percentile(coord_values, 95)
                mask = (coord_values >= p5) & (coord_values <= p95)
                filtered_coord_values = coord_values[mask]
                if len(filtered_coord_values) > 0:
                    filtered_offsets.append(np.mean(filtered_coord_values))
                else:
                    filtered_offsets.append(np.mean(coord_values))
            avg_offset = np.array(filtered_offsets, dtype=np.float64)

        offsets_map[body_name] = avg_offset

    # Ensure all nodes in mapping have an entry
    for node in node_mapping:
        if node not in offsets_map:
            offsets_map[node] = np.zeros(3, dtype=np.float64)

    # Log the computed offsets and some statistics
    logging.info(f"Subject {getattr(subject, 'ppid', 'unknown')} computed marker offsets from {len(calib_frames)} calibration frames:")
    for body_name, offset in offsets_map.items():
        if node_mapping.get(body_name) is not None:
            marker_name = node_mapping[body_name]
            offset_norm = np.linalg.norm(offset)
            num_frames_used = len(frame_offsets.get(body_name, []))
            logging.info(f"  {body_name} -> {marker_name}: offset=({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}) norm={offset_norm:.4f} m (from {num_frames_used} frames)")

    return offsets_map


def process_trial(trial, subject, node_mapping, offsets_map, base_skeleton, butter_cutoff=10, butter_order=2):
    if getattr(trial, "markers", None) is None:
        return None, None, None

    marker_timesteps = trial.markers
    num_frames = len(marker_timesteps)
    poses = np.zeros((base_skeleton.getNumDofs(), num_frames))
    fps = getattr(subject, "sampling_frequency", 120)
    trial_errors = []
    invalid_trial = False

    for frame_idx in range(num_frames):
        markers = []
        markerWeights = []
        frame_markers = marker_timesteps[frame_idx]
        targetPositions = []
        for body_name, marker_name in node_mapping.items():
            if marker_name is None:
                continue
            if marker_name not in frame_markers:
                continue
            try:
                body = base_skeleton.getBodyNode(body_name)
            except Exception:
                try:
                    body = base_skeleton.getBody(body_name)
                except Exception:
                    body = None
            if body is None:
                continue
            offset = offsets_map.get(body_name, np.zeros(3, dtype=np.float64))
            markers.append((body, np.asarray(offset, dtype=np.float64).reshape(3,)))
            markerWeights.append(1.0)

            # Read raw marker and convert + rotate to nimble frame (same as calibration)
            raw = np.asarray(frame_markers[marker_name], dtype=np.float64).reshape(3,)
            if np.linalg.norm(raw) > 10.0:
                raw_m = raw / 1000.0
            else:
                raw_m = raw
            tp = np.array([raw_m[0], raw_m[1], raw_m[2]], dtype=np.float64)
            targetPositions.append(tp)

        if len(targetPositions) == 0:
            if frame_idx == 0:
                try:
                    poses[:, frame_idx] = base_skeleton.getPositions()
                except Exception:
                    poses[:, frame_idx] = 0.0
            else:
                poses[:, frame_idx] = poses[:, frame_idx - 1]
            continue

        markerWeights = np.array(markerWeights, dtype=np.float64).reshape(-1, 1)
        targetPositions = np.array(targetPositions, dtype=np.float64).flatten()

        if frame_idx != 0:
            try:
                base_skeleton.setPositions(poses[:, frame_idx - 1])
            except Exception:
                pass

        try:
            error = base_skeleton.fitMarkersToWorldPositions(markers, targetPositions, markerWeights, scaleBodies=True)
        except Exception as e:
            logging.warning(f"fitMarkersToWorldPositions failed at frame {frame_idx}: {e}")
            invalid_trial = True
            break

        trial_errors.append(error)

        try:
            poses[:, frame_idx] = base_skeleton.getPositions()
        except Exception:
            logging.warning("Could not read positions after fit; marking trial invalid")
            invalid_trial = True
            break

        if np.any(np.isnan(poses[:, frame_idx])) or np.any(np.isinf(poses[:, frame_idx])):
            logging.warning(f"Invalid pose at frame {frame_idx}: NaN/Inf encountered")
            invalid_trial = True
            break

    if invalid_trial:
        return None, None, None

    if poses.shape[1] > butter_order * 2:
        nyq = 0.5 * fps
        normal_cutoff = butter_cutoff / nyq
        b, a = butter(butter_order, normal_cutoff, btype='low', analog=False)
        poses = filtfilt(b, a, poses, axis=1)

    marker_observations = [dict(frame_markers) for frame_markers in marker_timesteps]
    return poses, trial_errors, marker_observations


def process_subject(args):
    i, dataset, node_mapping, output_dir, rajagopal_osim_path = args
    subject = dataset[i]
    b3d_subject = nimble.biomechanics.SubjectOnDiskHeader()
    b3d_subject_pass = b3d_subject.addProcessingPass()
    b3d_subject_pass.setProcessingPassType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    base_model = nimble.RajagopalHumanBodyModel()
    base_skeleton = base_model.skeleton

    height = getattr(subject, "height", None)
    mass = getattr(subject, "mass", None)
    age = getattr(subject, "age", None)
    gender = getattr(subject, "gender", "unknown")
    sampling_frequency = getattr(subject, "sampling_frequency", None)
    if sampling_frequency is None or sampling_frequency <= 0:
        raise ValueError(f"Sampling frequency must be defined and > 0 for subject {getattr(subject, 'ppid', 'unknown')}")

    b3d_subject.setHeightM(height / 100.0 if height is not None else 0.0)
    b3d_subject.setMassKg(mass if mass is not None else 0.0)
    b3d_subject.setAgeYears(int(age) if age is not None else 0)
    b3d_subject.setBiologicalSex(str(gender).lower())
    b3d_subject.setNumDofs(base_skeleton.getNumDofs())
    b3d_subject.setNumJoints(base_skeleton.getNumJoints())
    b3d_subject.setNotes(get_subject_metadata(subject))

    # compute offsets using the calibration-based method (always use all frames)
    offsets_map = compute_offsets(subject, node_mapping, base_skeleton)

    errors = []
    trials = getattr(subject, "trials", [])
    num_trials = len(trials)

    # Collect average error per trial name for this subject
    errors_dict = {}

    for idx, trial in enumerate(trials):
        poses, trial_errors, marker_observations = process_trial(
            trial, subject, node_mapping, offsets_map, base_skeleton
        )
        if poses is None:
            continue
        b3d_trial = b3d_subject.addTrial()
        b3d_trial.setName(getattr(trial, "name", "unknown"))
        b3d_trial.setTrialLength(poses.shape[1])
        b3d_trial.setTimestep(1.0 / float(sampling_frequency))
        b3d_trial.setMarkerObservations(marker_observations)
        b3d_trial.setForcePlates([])
        trial_pass = b3d_trial.addPass()
        trial_pass.setType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
        trial_pass.setPoses(poses)
        # computeValues expects arrays for force/analog channels; provide empty placeholders
        num_frames = poses.shape[1]
        trial_pass.computeValues(
            base_skeleton,
            1.0 / float(sampling_frequency),
            poses,
            [],
            np.zeros((0, num_frames)),
            np.zeros((0, num_frames)),
            np.zeros((0, num_frames)),
            rootHistoryLen=10,
            rootHistoryStride=3)
        # attach Rajagopal osim text if available
        try:
            with open(rajagopal_osim_path, "r") as f:
                osim_content = f.read()
                b3d_subject_pass.setOpenSimFileText(osim_content)
        except Exception:
            pass

        trial_name = getattr(trial, "name", "unknown")
        if trial_errors is not None and len(trial_errors) > 0:
            avg_trial_error = np.mean(trial_errors)
            errors_dict[trial_name] = avg_trial_error
            logging.info(
                f"Error for subject {getattr(subject, 'ppid', 'unknown')}, trial {idx+1}/{num_trials} ({trial_name}): {avg_trial_error}"
            )
            errors.append(trial_errors)

    if len(errors) == 0:
        logging.warning(f"No valid trials for subject {getattr(subject, 'ppid', 'unknown')}, skipping save.")
        return getattr(subject, "ppid", "unknown"), {}

    all_errors = np.concatenate(errors) if len(errors) > 0 else np.array([])
    if all_errors.size > 0:
        overall_avg_error = np.mean(all_errors)
        logging.info(f"Overall average error for subject {getattr(subject, 'ppid', 'unknown')}: {overall_avg_error}")

    subject_name = getattr(subject, "ppid", "unknown")
    output_b3d_path = f"{output_dir}/{subject_name}.b3d"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_b3d_path):
        os.remove(output_b3d_path)
    nimble.biomechanics.SubjectOnDisk.writeB3D(output_b3d_path, b3d_subject)
    return subject_name, errors_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/mnt/datasets/183_subjects", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="/home/mnt/datasets/183_retargeted", help="Directory to save output .b3d files")
    parser.add_argument("--rajagopal_osim_path", type=str, default="/home/mnt/datasets/skeletons/rajagopal2015/Rajagopal2015.osim", help="Path to Rajagopal2015.osim file")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--indices", type=str, default=None, help="Inclusive subject index range 'start-end' (e.g., '90-95')")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    rajagopal_osim_path = args.rajagopal_osim_path
    num_workers = args.num_workers

    dataset = Subject183Dataset(dataset_path, preload=False)
    indices = range(len(dataset.ppids))
    # def parse_indices_range(spec: str, max_len: int):
    #     if spec is None:
    #         return range(max_len)
    #     spec = spec.strip()
    #     try:
    #         start_str, end_str = spec.split("-", 1)
    #         start = int(start_str.strip())
    #         end = int(end_str.strip())
    #     except Exception:
    #         raise ValueError(f"Invalid --indices '{spec}'. Use 'start-end', e.g., '90-95'.")
    #     if start < 0 or end < 0 or start >= max_len or end >= max_len:
    #         raise ValueError(f"--indices out of bounds 0..{max_len-1}: '{spec}'")
    #     if end < start:
    #         raise ValueError(f"--indices end < start: '{spec}'")
    #     return range(start, end + 1)

    # indices = parse_indices_range(args.indices, len(dataset.ppids))


    node_mapping = {
        "calcn_l":   "LHEE",
        "calcn_r":   "RHEE",
        "femur_l":   "LTHI",
        "femur_r":   "RTHI",
        "hand_l":    None,
        "hand_r":    None,
        "humerus_l": "LBIC",
        "humerus_r": "RBIC",
        "pelvis":    "LASI",
        "radius_l":  "LFOR",
        "radius_r":  "RFOR",
        "talus_l":   "LMED",
        "talus_r":   "RMED",
        "tibia_l":   None,
        "tibia_r":   "RSHA",
        "toes_l":    "LTOE",
        "toes_r":    "RTOE",
        "torso":     "STER",
        "ulna_l":    "LFOR",
        "ulna_r":    "RFOR",
    }

    arg_list = [
        (
            i,
            dataset,
            node_mapping,
            output_dir,
            rajagopal_osim_path
        )
        for i in indices
    ]

    all_trial_errors = {}
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_subject, arg_list), total=len(indices), desc="Processing subjects"))

    # Aggregate errors by trial name
    for _, trial_errors_by_name in results:
        for trial_name, avg_error in trial_errors_by_name.items():
            if trial_name not in all_trial_errors:
                all_trial_errors[trial_name] = []
            all_trial_errors[trial_name].append(avg_error)
    # Compute and log average error per trial type
    for trial_name, errors in all_trial_errors.items():
        trial_avg = np.mean(errors)
        logging.info(f"Global average error for trial '{trial_name}' across all subjects: {trial_avg}")
    # Optionally, compute overall global average error
    all_errors_flat = [err for errors in all_trial_errors.values() for err in errors]
    if all_errors_flat:
        global_avg_error = np.mean(all_errors_flat)
        logging.info(f"Global average error across all subjects and trials: {global_avg_error}")

if __name__ == "__main__":
    main()