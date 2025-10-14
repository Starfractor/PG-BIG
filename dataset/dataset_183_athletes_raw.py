import ezc3d
import glob
import os
import pandas as pd
import re
from torch.utils.data import Dataset

class Trial183:
    def __init__(self, name, c3d_path=None):
        self.name = name
        self.markers = None
        self.c3d_path = c3d_path

        if c3d_path is not None:
            c3d = ezc3d.c3d(c3d_path)
            labels = c3d['parameters']['POINT']['LABELS']['value']
            labels = [l.decode() if isinstance(l, bytes) else l for l in labels]
            points = c3d['data']['points']  # shape: (3, n_markers, n_frames)
            n_frames = points.shape[2]
            self.markers = [
                [
                    {"label": label, "index": i, "value": points[:, i, frame_idx]}
                    for i, label in enumerate(labels)
                ]
                for frame_idx in range(n_frames)
            ]

class Athlete183:
    def __init__(
        self,
        ppid: int,
        base_directory: str,
        subject_info: dict
    ):
        self.ppid = str(ppid)
        self.trials = []
        self.calibration = None
        self.sampling_frequency = subject_info.get("sampling_frequency")
        self.gender = subject_info.get("gender")
        self.age = subject_info.get("age")
        self.height = subject_info.get("height")
        self.mass = subject_info.get("mass")
        self.level = subject_info.get("level")
        self.sport = subject_info.get("sport")
        self.injuries = subject_info.get("injuries") if subject_info.get("injuries") is not None else []

        # Load C3D trials (only file paths, not full data)
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
            if trial_name == "calib":
                self.calibration = c3d_file
            else:
                trial_name = os.path.splitext(trial_name)[0]
                self.trials.append((trial_name, c3d_file))

class Athlete183RawDataset(Dataset):
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        self.subject_info = self.get_subject_info()
        self.ppids = list(self.subject_info.keys())

    def get_subject_info(self):
        kinematic_directory = os.path.join(self.base_directory, "Kinematic_Data")
        if not os.path.exists(kinematic_directory):
            return {}
        ppids = sorted([
            d for d in os.listdir(kinematic_directory)
            if os.path.isdir(os.path.join(kinematic_directory, d)) and d.isdigit()
        ], key=lambda x: int(x))

        # Load metadata once
        subject_log_path = os.path.join(self.base_directory, "Participants Info", "Subject Log.xlsx")
        metadata = {}
        if os.path.exists(subject_log_path):
            df = pd.read_excel(subject_log_path)
            df.columns = df.columns.str.strip()
            subj_id_col = "Subject ID"
            df[subj_id_col] = df[subj_id_col].astype(str).str.strip().str.lower()
            for _, row in df.iterrows():
                ppid_str = str(row[subj_id_col]).strip().lower()
                metadata[ppid_str] = row

        # Load sampling frequency once
        sampling_freq_path = os.path.join(self.base_directory, "Participants Info", "Sampling Frequency.xlsx")
        freq = {}
        if os.path.exists(sampling_freq_path):
            df = pd.read_excel(sampling_freq_path)
            df.columns = df.columns.str.strip()
            df["PPID"] = df["PPID"].astype(str).str.strip().str.lower()
            for _, row in df.iterrows():
                ppid_str = str(row["PPID"]).strip().lower()
                freq[ppid_str] = row.get("Sampling Frequency (Hz)", None)

        # Build subject info dictionary with all basic info preloaded
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
        return subject_info

    def __len__(self) -> int:
        return len(self.ppids)

    def __getitem__(self, idx: int) -> Athlete183:
        ppid = self.ppids[idx]
        info = self.subject_info[ppid]
        # Only load metadata and file paths here, not full marker data
        return Athlete183(
            ppid,
            self.base_directory,
            subject_info=info
        )