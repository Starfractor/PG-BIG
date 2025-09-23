import argparse
import os
import glob
import numpy as np
import nimblephysics as nimble
from tqdm import tqdm


def retarget_and_fill_trial(
    motion,
    base_skeleton,
    b3d_trial,
    trial_name="retargeted_motion",
    timestep=1.0/30.0
):
    # Define mapping from your motion data to Rajagopal DOFs
    mapping_bodyJoints = {
        'hip_l': 1,  # Left hip
        'hip_r': 2,  # Right hip
        'walker_knee_l': 4,  # Left knee
        'walker_knee_r': 5,  # Right knee
        'ankle_l': 7,  # Left ankle
        'ankle_r': 8,  # Right ankle
        'mtp_l': 10,  # Left mtp
        'mtp_r': 11,  # Right mtp
        'acromial_l': 16,  # Left Shoulder
        'acromial_r': 17,  # Right shoulder
        'elbow_l': 18,  # Left elbow
        'elbow_r': 19,  # Right elbow
        'radius_hand_l': 20,  # Left wrist
        'radius_hand_r': 21,  # Right wrist
    }

    osim_joint_names = [
        'ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r',
        'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back',
        'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r',
        'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l'
    ]
    osim_dict = {name: i for i, name in enumerate(osim_joint_names)}

    osim_index = np.array([osim_dict[name] for name in mapping_bodyJoints])
    smpl_index = np.array([mapping_bodyJoints[name] for name in mapping_bodyJoints])

    num_frames = motion.shape[0]
    num_dofs = base_skeleton.getNumDofs()
    poses = np.zeros((num_dofs, num_frames))

    # Map the motion data columns to the correct Rajagopal DOFs
    if motion.shape[1] >= smpl_index.max() + 1:
        for i, model_idx in enumerate(osim_index):
            poses[model_idx, :] = motion[:, smpl_index[i], 0]
    else:
        raise ValueError(f"Motion shape {motion.shape} does not match mapping indices (max index {smpl_index.max()})")

    # Prepare joint mapping for retargeting
    bodyJoints = [base_skeleton.getJoint(name) for name in mapping_bodyJoints]
    target_joints_indices = [mapping_bodyJoints[name] for name in mapping_bodyJoints]

    # Retarget each frame using fitJointsToWorldPositions
    errors = []
    for t in range(num_frames):
        target_joints = motion[t, target_joints_indices, :].astype(np.float64).reshape((-1, 1))
        if t == 0:
            base_skeleton.setPositions(np.zeros(num_dofs))
        else:
            base_skeleton.setPositions(poses[:, t-1])
        error = base_skeleton.fitJointsToWorldPositions(bodyJoints, target_joints, scaleBodies=True, logOutput=False, lineSearch=True)
        poses[:, t] = base_skeleton.getPositions()
        errors.append(error)

    # Fill the provided b3d_trial object
    b3d_trial.setName(trial_name)
    b3d_trial.setTrialLength(num_frames)
    b3d_trial.setTimestep(timestep)
    b3d_trial.setMarkerObservations([{} for _ in range(num_frames)])
    b3d_trial.setForcePlates([])

    trial_pass = b3d_trial.addPass()
    trial_pass.setType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    trial_pass.setPoses(poses)
    trial_pass.computeValues(
        base_skeleton,
        timestep,
        poses,
        [],
        np.zeros((0, num_frames)),
        np.zeros((0, num_frames)),
        np.zeros((0, num_frames)),
        rootHistoryLen=10,
        rootHistoryStride=3
    )

    return np.mean(errors)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_dir", 
        type=str, 
        default="/home/mnt/Generations/Default/MDM", 
        help="Define where your motion files that will be retargeted are."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='mdm',
        choices=['mdm', 'persona_booth', 't2m-gpt'],
        help="Specify the baseline: \"mdm\", \"persona_booth\", or \"t2m-gpt\""
    )
    parser.add_argument("--output_dir", 
        type=str, 
        default='output', 
        help='Define where to save the retargeted motions.')
    parser.add_argument(
        "--subject_ref", 
        type=str, 
        default=None, 
        help='output directory'
    )
    args = parser.parse_args()

    if os.path.isdir(args.file_dir):
        glob_files = glob.glob(os.path.join(args.file_dir, "**", "*.npy"), recursive=True)
    else:
        glob_files = glob.glob(args.file_dir)
    print(f'Found {len(glob_files)} files')

    all_motions = []
    for file in glob_files:
        if args.dataset == 'mdm':
            data = np.load(file, allow_pickle=True).item()
            motions = data['motion'].transpose(0, 3, 1, 2)
            timestep = 1.0 / 20
            for motion in motions:
                if motion.shape[0] == 1:
                    motion = np.squeeze(motion, axis=0)
                all_motions.append((motion, timestep))
        elif args.dataset == "persona_booth":
            data = np.load(file, allow_pickle=True).item()
            motions = data['motion'].transpose(0, 3, 1, 2)
            if motions.shape[0] == 1:
                motions = np.squeeze(motions, axis=0)
            timestep = 1.0 / 20.0
            for motion in motions:
                all_motions.append((motion, timestep))
        elif args.dataset == "t2m-gpt":
            motion = np.load(file)
            if motion.shape[0] == 1:
                motion = np.squeeze(motion, axis=0)
            timestep = 1.0 / 20.0
            all_motions.append((motion, timestep))
        else:
            raise ValueError(f'Invalid dataset type: {args.dataset}')

    if not all_motions:
        print("No motions found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Create subject and processing pass once
    base_model = nimble.RajagopalHumanBodyModel()
    base_skeleton = base_model.skeleton
    b3d_subject = nimble.biomechanics.SubjectOnDiskHeader()
    b3d_subject_pass = b3d_subject.addProcessingPass()
    b3d_subject_pass.setProcessingPassType(nimble.biomechanics.ProcessingPassType.KINEMATICS)
    b3d_subject.setHeightM(0)
    b3d_subject.setMassKg(0)
    b3d_subject.setAgeYears(0)
    b3d_subject.setBiologicalSex("unknown")
    b3d_subject.setNumDofs(base_skeleton.getNumDofs())
    b3d_subject.setNumJoints(base_skeleton.getNumJoints())

    # Optionally embed OpenSim file text
    rajagopal_osim_path = "/home/mnt/datasets/skeletons/rajagopal2015/Rajagopal2015.osim"
    if os.path.exists(rajagopal_osim_path):
        with open(rajagopal_osim_path, "r") as f:
            osim_content = f.read()
            b3d_subject_pass.setOpenSimFileText(osim_content)

    # Add all trials
    errors = []
    for idx, (motion, timestep) in enumerate(tqdm(all_motions, desc="Retargeting motions")):
        b3d_trial = b3d_subject.addTrial()
        error = retarget_and_fill_trial(
            motion,
            base_skeleton,
            b3d_trial,
            trial_name=f"motion_{idx}",
            timestep=timestep
        )
        errors.append(error)

    if errors:
        print(f"Average retargeting error across all trials: {np.mean(errors):.6f}")
        for i, err in enumerate(errors):
            print(f"  Trial {i}: {err:.6f}")

    # Save .b3d once
    b3d_path = os.path.join(args.output_dir, f"{args.dataset}_motions.b3d")
    if os.path.exists(b3d_path):
        os.remove(b3d_path)
    nimble.biomechanics.SubjectOnDisk.writeB3D(b3d_path, b3d_subject)
    print(f"Saved b3d file to: {b3d_path}")

if __name__ == "__main__":
    main()