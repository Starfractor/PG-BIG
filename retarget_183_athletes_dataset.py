import argparse
import json
import nimblephysics as nimble
import numpy as np
import os
import logging
import ezc3d
import glob
import pandas as pd
import re
from scipy.signal import butter, filtfilt
from multiprocessing import Pool
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# For a few participants some trials were split into left/right files but should
# be treated as the same class. Map participant id (string) -> set of base names
# that should be collapsed (i.e. ankle_left -> ankle).
COLLAPSE_TRIALS_PER_SUBJECT = {
    '1424': {'scorpion', 'ankle'},
    '1913': {'ankle'},
}

def load_c3d_trial(c3d_path):
    """Load a single C3D file and return markers data."""
    c3d = ezc3d.c3d(c3d_path)
    labels = c3d['parameters']['POINT']['LABELS']['value']
    labels = [l.decode() if isinstance(l, bytes) else l for l in labels]
    points = c3d['data']['points']  # shape: (4, n_markers, n_frames) - 4th dim is confidence/residual
    n_frames = points.shape[2]
    
    markers = [
        [
            {"label": label, "index": i, "value": points[:3, i, frame_idx]}  # Only take first 3 dims (x,y,z)
            for i, label in enumerate(labels)
        ]
        for frame_idx in range(n_frames)
    ]
    return markers

def load_subject_trials(ppid, base_directory):
    """Load all trials for a subject and return trial info."""
    trials = []
    calibration = None
    
    c3d_folder = os.path.join(base_directory, "Kinematic_Data", str(ppid), "Generated_C3D_files")
    c3d_files = sorted(glob.glob(os.path.join(c3d_folder, "*.c3d")))
    
    for c3d_file in c3d_files:
        trial_name = os.path.basename(c3d_file).lower()
        trial_name = trial_name.replace('.c3d', '')
        trial_name = re.sub(r'[0-9]+', '', trial_name)
        trial_name = re.sub(r'^_+|_+$', '', trial_name)
        trial_name = re.sub(r'_[0-9]+$', '', trial_name)
        trial_name = trial_name.strip().lower().replace(" ", "_").replace("-", "_")
        trial_name = '_'.join([w[:-1] if w.endswith('s') else w for w in trial_name.split('_')])
        # Fix typos within dataset
        trial_name = (
            trial_name
            .replace('rotatry', 'rotary')
            .replace('rotarty', 'rotary')
            .replace('scoprion', 'scorpion')
            .replace('cros_adduction', 'crossover_adduction')
        )
        # collapse repeated underscores and strip leading/trailing underscores
        trial_name = re.sub(r'_+', '_', trial_name).strip('_')
        
        if trial_name == "calib":
            calibration = {
                'name': trial_name,
                'path': c3d_file,
                'markers': None  # Load lazily when needed
            }
        else:
            trial_name = os.path.splitext(trial_name)[0]
            trials.append({
                'name': trial_name,
                'path': c3d_file,
                'markers': None  # Load lazily when needed
            })
    
    # Sort trials alphabetically by normalized name so order is deterministic
    try:
        trials.sort(key=lambda t: re.sub(r'_+', '_', t['name']).lower())
    except Exception:
        # fallback: sort by raw name if normalization fails for some reason
        trials.sort(key=lambda t: t['name'].lower() if isinstance(t.get('name'), str) else '')

    # After sorting, collapse left/right variants for exceptional participants
    try:
        collapse_set = COLLAPSE_TRIALS_PER_SUBJECT.get(str(ppid), set())
        if collapse_set:
            for t in trials:
                m = re.match(r'^(.*)_(left|right)$', t['name'])
                if m and m.group(1) in collapse_set:
                    # preserve order, just canonicalize the name to base
                    t['name'] = m.group(1)
    except Exception:
        pass

    return trials, calibration

def load_subject_info(base_directory):
    """Load subject metadata and sampling frequency information."""
    kinematic_directory = os.path.join(base_directory, "Kinematic_Data")
    if not os.path.exists(kinematic_directory):
        return {}, []
    
    ppids = sorted([
        d for d in os.listdir(kinematic_directory)
        if os.path.isdir(os.path.join(kinematic_directory, d)) and d.isdigit()
    ], key=lambda x: int(x))

    # Load metadata
    subject_log_path = os.path.join(base_directory, "Participants Info", "Subject Log.xlsx")
    metadata = {}
    if os.path.exists(subject_log_path):
        df = pd.read_excel(subject_log_path)
        df.columns = df.columns.str.strip()
        subj_id_col = "Subject ID"
        df[subj_id_col] = df[subj_id_col].astype(str).str.strip().str.lower()
        for _, row in df.iterrows():
            ppid_str = str(row[subj_id_col]).strip().lower()
            metadata[ppid_str] = row

    # Load sampling frequency
    sampling_freq_path = os.path.join(base_directory, "Participants Info", "Sampling Frequency.xlsx")
    freq = {}
    if os.path.exists(sampling_freq_path):
        df = pd.read_excel(sampling_freq_path)
        df.columns = df.columns.str.strip()
        df["PPID"] = df["PPID"].astype(str).str.strip().str.lower()
        for _, row in df.iterrows():
            ppid_str = str(row["PPID"]).strip().lower()
            freq[ppid_str] = row.get("Sampling Frequency (Hz)", None)

    # Build subject info dictionary
    subject_info = {}
    for ppid in ppids:
        ppid_str = str(ppid).strip().lower()
        meta = metadata.get(ppid_str, None)
        info = {
            "ppid": ppid,
            "age": meta.get("Age") if meta is not None else None,
            "gender": meta.get("Gender") if meta is not None else None,
            "height": meta.get("Height (cm)") if meta is not None else None,
            "mass": meta.get("Mass (kg)") if meta is not None else None,
            "level": meta.get("Level") if meta is not None else None,
            "sport": meta.get("Sport") if meta is not None else None,
            "injuries": [meta.get(f"Injury {i}") for i in range(1, 6)] if meta is not None else [],
            "sampling_frequency": freq.get(ppid_str, None)
        }
        subject_info[ppid] = info
    return subject_info, ppids

def get_subject_metadata(subject_info):
    """Convert subject info to JSON metadata string."""
    notes_dict = {
        "sport": subject_info.get("sport", "unknown"),
        "level": subject_info.get("level", "unknown"),
        "injuries": subject_info.get("injuries", []),
    }
    return json.dumps(notes_dict)

def build_label_indices(labels):
    """Build a mapping from marker label to all indices (for repeated labels)."""
    label_indices = {}
    for i, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(i)
    return label_indices

def get_marker_index(label_indices, marker_label, anatomical_idx=0):
    """Get the correct marker index for a repeated label, using anatomical order index."""
    indices = label_indices.get(marker_label, [])
    if len(indices) > anatomical_idx:
        return indices[anatomical_idx]
    return None

def compute_offsets(calibration, node_mapping, base_skeleton, ppid):
    """Compute marker offsets from calibration data."""
    offsets_map = {}

    if calibration is None:
        logging.warning(f"Subject {ppid} has no calibration; using zero offsets")
        for node in node_mapping:
            offsets_map[node] = np.zeros(3, dtype=np.float64)
        return offsets_map

    # Load calibration markers if not already loaded
    if calibration['markers'] is None:
        calibration['markers'] = load_c3d_trial(calibration['path'])
    
    calib_frames = calibration['markers']
    if len(calib_frames) == 0:
        logging.warning(f"No calibration frames for subject {ppid}; using zero offsets")
        for node in node_mapping:
            offsets_map[node] = np.zeros(3, dtype=np.float64)
        return offsets_map

    # Build label_indices from the first calibration frame
    first_frame_labels = [m['label'] for m in calib_frames[0]]
    label_indices = build_label_indices(first_frame_labels)

    frame_offsets = {}
    for frame_idx, frame_markers in enumerate(calib_frames):
        markers_for_fit = []
        target_positions_list = []
        weights = []
        processed_node_names = []

        for body_name, (marker_label, anatomical_idx) in node_mapping.items():
            if marker_label is None:
                continue

            marker_idx = get_marker_index(label_indices, marker_label, anatomical_idx)
            if marker_idx is None:
                continue

            if marker_idx >= len(frame_markers):
                continue
            marker_info = frame_markers[marker_idx]
            coord = marker_info['value']
            if coord is None:
                continue
            coord_arr = np.asarray(coord, dtype=float)[:3]  # Take only first 3 elements (x,y,z)
            if coord_arr.size < 3 or np.all(np.isnan(coord_arr)) or np.allclose(coord_arr, 0.0):
                continue

            body = None
            try:
                body = base_skeleton.getBodyNode(body_name)
            except Exception:
                try:
                    body = base_skeleton.getBody(body_name)
                except Exception:
                    body = None
            if body is None:
                continue

            offset_vec = np.zeros(3, dtype=np.float64)
            markers_for_fit.append((body, offset_vec))

            if np.linalg.norm(coord_arr) > 10.0:
                coord_m = coord_arr / 1000.0
            else:
                coord_m = coord_arr
            tp = np.array([coord_m[0], coord_m[1], coord_m[2]], dtype=np.float64).reshape(3,)
            target_positions_list.append(tp)
            weights.append(1.0)
            processed_node_names.append((body_name, marker_label))

        if len(target_positions_list) == 0:
            continue

        target_positions_mat = np.vstack(target_positions_list).astype(np.float64)
        target_positions_flat = target_positions_mat.flatten()
        marker_weights_np = np.array(weights, dtype=np.float64).reshape(-1, 1)

        try:
            base_skeleton.fitMarkersToWorldPositions(markers_for_fit, target_positions_flat, marker_weights_np, scaleBodies=False)
        except Exception:
            continue

        for (body_name, marker_label), (body, _) in zip(processed_node_names, markers_for_fit):
            world_pos = target_positions_list[processed_node_names.index((body_name, marker_label))]
            world_tf = body.getWorldTransform()
            R = None
            t = None
            try:
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
                continue
            local_offset_np = R.T.dot(world_pos - t)
            if body_name not in frame_offsets:
                frame_offsets[body_name] = []
            frame_offsets[body_name].append(local_offset_np.copy())

    # Compute average offsets
    for body_name in node_mapping.keys():
        if body_name not in frame_offsets or len(frame_offsets[body_name]) == 0:
            offsets_map[body_name] = np.zeros(3, dtype=np.float64)
            continue
        offset_vectors = np.array(frame_offsets[body_name])
        if offset_vectors.shape[0] < 10:
            avg_offset = np.mean(offset_vectors, axis=0)
        else:
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

    for node in node_mapping:
        if node not in offsets_map:
            offsets_map[node] = np.zeros(3, dtype=np.float64)

    logging.info(f"Subject {ppid} computed marker offsets from {len(calib_frames)} calibration frames:")
    for body_name, (marker_label, anatomical_idx) in node_mapping.items():
        if marker_label is not None:
            offset = offsets_map[body_name]
            offset_norm = np.linalg.norm(offset)
            num_frames_used = len(frame_offsets.get(body_name, []))
            logging.info(f"  {body_name} -> {marker_label}[{anatomical_idx}]: offset=({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}) norm={offset_norm:.4f} m (from {num_frames_used} frames)")

    return offsets_map

def process_trial(trial, node_mapping, offsets_map, base_skeleton, sampling_frequency, butter_cutoff=10, butter_order=2):
    """Process a single trial and return poses, errors, and marker observations."""
    # Load trial markers if not already loaded
    if trial['markers'] is None:
        trial['markers'] = load_c3d_trial(trial['path'])
    
    marker_timesteps = trial['markers']
    num_frames = len(marker_timesteps)
    poses = np.zeros((base_skeleton.getNumDofs(), num_frames))
    trial_errors = []
    invalid_trial = False

    # Build label_indices from the first frame
    first_frame_labels = [m['label'] for m in marker_timesteps[0]]
    label_indices = build_label_indices(first_frame_labels)

    # Initialize skeleton to neutral pose before processing
    try:
        base_skeleton.setPositions(np.zeros(base_skeleton.getNumDofs()))
    except Exception:
        pass

    for frame_idx in range(num_frames):
        markers = []
        markerWeights = []
        frame_markers = marker_timesteps[frame_idx]
        targetPositions = []
        
        for body_name, (marker_label, anatomical_idx) in node_mapping.items():
            if marker_label is None:
                continue

            marker_idx = get_marker_index(label_indices, marker_label, anatomical_idx)
            if marker_idx is None:
                continue
            if marker_idx >= len(frame_markers):
                continue
            marker_info = frame_markers[marker_idx]
            
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
            
            raw = np.asarray(marker_info['value'], dtype=np.float64)[:3]  # Take only first 3 elements (x,y,z)
            
            # Check for invalid marker data
            if np.any(np.isnan(raw)) or np.any(np.isinf(raw)):
                continue
            
            # Check if marker is at origin (likely invalid)
            if np.allclose(raw, 0.0):
                continue
            
            if np.linalg.norm(raw) > 10.0:
                raw_m = raw / 1000.0
            else:
                raw_m = raw
            
            tp = np.array([raw_m[0], raw_m[1], raw_m[2]], dtype=np.float64)
            
            markers.append((body, np.asarray(offset, dtype=np.float64).reshape(3,)))
            markerWeights.append(1.0)
            targetPositions.append(tp)

        if len(targetPositions) == 0:
            logging.warning(f"Trial {trial['name']}, frame {frame_idx}: No valid markers found, skipping trial")
            invalid_trial = True
            break

        markerWeights = np.array(markerWeights, dtype=np.float64).reshape(-1, 1)
        targetPositions = np.array(targetPositions, dtype=np.float64).flatten()

        # Check for invalid target positions before fitting
        if np.any(np.isnan(targetPositions)) or np.any(np.isinf(targetPositions)):
            logging.warning(f"Trial {trial['name']}, frame {frame_idx}: Invalid target positions (NaN/Inf), skipping trial")
            invalid_trial = True
            break

        if frame_idx != 0:
            try:
                base_skeleton.setPositions(poses[:, frame_idx - 1])
            except Exception:
                pass

        try:
            error = base_skeleton.fitMarkersToWorldPositions(markers, targetPositions, markerWeights, scaleBodies=True)
        except Exception as e:
            logging.warning(f"Trial {trial['name']}, frame {frame_idx}: fitMarkersToWorldPositions failed: {e}")
            invalid_trial = True
            break

        trial_errors.append(error)

        try:
            poses[:, frame_idx] = base_skeleton.getPositions()
        except Exception:
            logging.warning(f"Trial {trial['name']}, frame {frame_idx}: Could not read positions after fit")
            invalid_trial = True
            break

        if np.any(np.isnan(poses[:, frame_idx])) or np.any(np.isinf(poses[:, frame_idx])):
            logging.warning(f"Trial {trial['name']}, frame {frame_idx}: Invalid pose (NaN/Inf) after fitting with {len(markers)} markers")
            invalid_trial = True
            break

    if invalid_trial:
        return None, None, None

    if poses.shape[1] > butter_order * 2:
        nyq = 0.5 * sampling_frequency
        normal_cutoff = butter_cutoff / nyq
        b, a = butter(butter_order, normal_cutoff, btype='low', analog=False)
        poses = filtfilt(b, a, poses, axis=1)

    # marker_observations: list of dicts with all marker info per frame
    marker_observations = [
        {m['label']: m['value'] for m in frame_markers}
        for frame_markers in marker_timesteps
    ]
    return poses, trial_errors, marker_observations

def process_subject(args):
    """Process a single subject and save to B3D format."""
    ppid, base_directory, subject_info, node_mapping, output_dir, rajagopal_osim_path = args
    
    # Load subject data
    info = subject_info[ppid]
    trials, calibration = load_subject_trials(ppid, base_directory)
    
    b3d_subject = nimble.biomechanics.SubjectOnDiskHeader()
    b3d_subject_pass = b3d_subject.addProcessingPass()
    b3d_subject_pass.setProcessingPassType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    base_model = nimble.RajagopalHumanBodyModel()
    base_skeleton = base_model.skeleton

    # Apply coordinate system transformation to skeleton
    root_joint = base_skeleton.getRootJoint()
    if root_joint is not None:
        # Create rotation matrix: [π/2, 0, 0] = 90° around X-axis
        rotation_matrix = nimble.math.eulerXYZToMatrix(np.array([np.pi/2, 0, 0]))
        
        # Get current transform
        current_transform = root_joint.getTransformFromParentBodyNode()
        
        # Apply rotation to the skeleton's coordinate frame
        new_transform = nimble.math.Isometry3()
        new_transform.set_rotation(rotation_matrix @ current_transform.rotation())
        new_transform.set_translation(current_transform.translation())
        root_joint.setTransformFromParentBodyNode(new_transform)

    height = info.get("height")
    mass = info.get("mass")
    age = info.get("age")
    gender = info.get("gender", "unknown")
    sampling_frequency = info.get("sampling_frequency")
    
    if sampling_frequency is None or sampling_frequency <= 0:
        raise ValueError(f"Sampling frequency must be defined and > 0 for subject {ppid}")

    b3d_subject.setHeightM(height / 100.0 if height is not None else 0.0)
    b3d_subject.setMassKg(mass if mass is not None else 0.0)
    b3d_subject.setAgeYears(int(age) if age is not None else 0)
    b3d_subject.setBiologicalSex(str(gender).lower())
    b3d_subject.setNumDofs(base_skeleton.getNumDofs())
    b3d_subject.setNumJoints(base_skeleton.getNumJoints())
    b3d_subject.setNotes(get_subject_metadata(info))

    # Compute offsets
    offsets_map = compute_offsets(calibration, node_mapping, base_skeleton, ppid)

    errors = []
    errors_dict = {}

    for idx, trial in enumerate(trials):
        poses, trial_errors, marker_observations = process_trial(
            trial, node_mapping, offsets_map, base_skeleton, sampling_frequency
        )
        if poses is None:
            continue
            
        b3d_trial = b3d_subject.addTrial()
        b3d_trial.setName(trial['name'])
        b3d_trial.setTrialLength(poses.shape[1])
        b3d_trial.setTimestep(1.0 / float(sampling_frequency))
        b3d_trial.setMarkerObservations(marker_observations)
        b3d_trial.setForcePlates([])
        trial_pass = b3d_trial.addPass()
        trial_pass.setType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
        trial_pass.setPoses(poses)
        
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
            
        # Attach Rajagopal osim text if available
        try:
            with open(rajagopal_osim_path, "r") as f:
                osim_content = f.read()
                b3d_subject_pass.setOpenSimFileText(osim_content)
        except Exception:
            pass

        trial_name = trial['name']
        if trial_errors is not None and len(trial_errors) > 0:
            avg_trial_error = np.mean(trial_errors)
            errors_dict[trial_name] = avg_trial_error
            logging.info(f"Error for subject {ppid}, trial {idx+1}/{len(trials)} ({trial_name}): {avg_trial_error}")
            errors.append(trial_errors)

    if len(errors) == 0:
        logging.warning(f"No valid trials for subject {ppid}, skipping save.")
        return ppid, {}

    all_errors = np.concatenate(errors) if len(errors) > 0 else np.array([])
    if all_errors.size > 0:
        overall_avg_error = np.mean(all_errors)
        logging.info(f"Overall average error for subject {ppid}: {overall_avg_error}")

    output_b3d_path = f"{output_dir}/{ppid}.b3d"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_b3d_path):
        os.remove(output_b3d_path)
    nimble.biomechanics.SubjectOnDisk.writeB3D(output_b3d_path, b3d_subject)
    return ppid, errors_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/mnt/datasets/183_subjects", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="/home/mnt/datasets/183_retargeted", help="Directory to save output .b3d files")
    parser.add_argument("--rajagopal_osim_path", type=str, default="/home/mnt/datasets/skeletons/rajagopal2015/Rajagopal2015.osim", help="Path to Rajagopal2015.osim file")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--indices", type=str, default=None, help="Inclusive subject index range 'start-end' (e.g., '90-95')")

    args = parser.parse_args()

    # Load subject info directly
    subject_info, ppids = load_subject_info(args.dataset_path)

    # Specify which subjects to retarget
    if args.indices is None or str(args.indices).lower() == "none":
        selected_ppids = ppids
    else:
        spec = str(args.indices).strip()
        try:
            start_str, end_str = spec.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
        except Exception:
            raise ValueError(f"Invalid --indices '{spec}'. Use 'start-end', e.g., '90-95'.")
        if start < 0 or end < 0 or start >= len(ppids) or end >= len(ppids):
            raise ValueError(f"--indices out of bounds 0..{len(ppids)-1}: '{spec}'")
        if end < start:
            raise ValueError(f"--indices end < start: '{spec}'")
        selected_ppids = ppids[start:end+1]

    # Node mapping: body_name -> (marker_label, anatomical_index)
    # anatomical_index specifies which occurrence of the marker to use (0 = first, 1 = second, etc.)
    node_mapping = {
        "calcn_l":   ("LHEE", 0), # calcn_; - 
        "calcn_r":   ("RHEE", 0),
        "femur_l":   ("LTHI", 0),
        "femur_r":   ("RTHI", 0),
        "hand_l":    ("LMED", 1),   # wrist - second LMED
        "hand_r":    ("RMED", 1),   # wrist - second RMED
        "humerus_l": ("LBIC", 0),
        "humerus_r": ("RBIC", 0),
        "pelvis":    ("LASI", 0),
        "radius_l":  (None, 0),
        "radius_r":  (None, 0),
        "talus_l":   ("LMED", 3),   # ankle - fourth LMED
        "talus_r":   ("RMED", 3),   # ankle - fourth RMED
        "tibia_l":   (None, 0),     # No 
        "tibia_r":   ("RSHA", 0),   #
        "toes_l":    ("LTOE", 0),
        "toes_r":    ("RTOE", 0),
        "torso":     ("STER", 0),
        "ulna_l":    ("LFOR", 0),
        "ulna_r":    ("RFOR", 0),
    }

    arg_list = [
        (
            ppid,
            args.dataset_path,
            subject_info,
            node_mapping,
            args.output_dir,
            args.rajagopal_osim_path
        )
        for ppid in selected_ppids
    ]

    all_trial_errors = {}
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_subject, arg_list), total=len(selected_ppids), desc="Processing subjects"))

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
        
    # Compute overall global average error
    all_errors_flat = [err for errors in all_trial_errors.values() for err in errors]
    if all_errors_flat:
        global_avg_error = np.mean(all_errors_flat)
        logging.info(f"Global average error across all subjects and trials: {global_avg_error}")

if __name__ == "__main__":
    main()