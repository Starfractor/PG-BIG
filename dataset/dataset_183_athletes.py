import glob
import nimblephysics as nimble
import os
import pandas as pd
import re
import sys
from contextlib import contextmanager
from torch.utils.data import Dataset
from tqdm import tqdm


@contextmanager
def suppress_output():
    devnull = open(os.devnull, "w")
    try:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(orig_stdout_fd, 1)
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    finally:
        devnull.close()

class Trial183:
    def __init__(self, name, c3d_path=None):
        self.name = name
        self.markers = None
        self.force_plates = None
        self.c3d_path = c3d_path

        if c3d_path is not None:
            with suppress_output():
                c3d_data = nimble.biomechanics.C3DLoader.loadC3D(c3d_path)
            self.markers = c3d_data.markerTimesteps
            self.force_plates = c3d_data.forcePlates

class Subject183:
    def __init__(
        self,
        ppid: int,
        base_directory: str,
        gender=None,
        age=None,
        height=None,
        mass=None,
        level=None,
        sport=None,
        injuries=None,
        sampling_frequency=None
    ):
        self.ppid = str(ppid)
        self.trials = []
        self.calibration = None
        self.sampling_frequency = sampling_frequency
        self.gender = gender
        self.age = age
        self.height = height
        self.mass = mass
        self.level = level
        self.sport = sport
        self.injuries = injuries if injuries is not None else []

        # Load C3D trials 
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
                self.calibration = Trial183(name="calibration", c3d_path=c3d_file)
            else:
                trial_name = os.path.splitext(trial_name)[0]
                self.trials.append(Trial183(name=trial_name, c3d_path=c3d_file))

class Subject183Dataset(Dataset):
    def __init__(self, base_directory: str, preload: bool = False):
        self.base_directory = base_directory
        self.subject_info = self.get_subject_info()
        self.ppids = list(self.subject_info.keys())
        self.preload = preload
        self._subjects = None
        if self.preload:
            self._subjects = []
            for ppid in tqdm(self.ppids, desc="Preloading subjects"):
                info = self.subject_info[ppid]
                self._subjects.append(
                    Subject183(
                        ppid,
                        self.base_directory,
                        gender=info.get("gender"),
                        age=info.get("age"),
                        height=info.get("height"),
                        mass=info.get("mass"),
                        level=info.get("level"),
                        sport=info.get("sport"),
                        injuries=info.get("injuries"),
                        sampling_frequency=info.get("sampling_frequency")
                    )
                )

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

    def __getitem__(self, idx: int) -> Subject183:
        ppid = self.ppids[idx]
        info = self.subject_info[ppid]
        if self.preload and self._subjects is not None:
            return self._subjects[idx]
        else:
            return Subject183(
                ppid,
                self.base_directory,
                gender=info.get("gender"),
                age=info.get("age"),
                height=info.get("height"),
                mass=info.get("mass"),
                level=info.get("level"),
                sport=info.get("sport"),
                injuries=info.get("injuries"),
                sampling_frequency=info.get("sampling_frequency")
            )

def load_trial(c3d_file_path):
    raw_name = os.path.basename(c3d_file_path)
    name = raw_name.replace('.c3d', '')
    name = re.sub(r'[0-9]+', '', name)
    name = re.sub(r'^_+|_+$', '', name)
    name = re.sub(r'_[0-9]+$', '', name)
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    name = '_'.join([w[:-1] if w.endswith('s') else w for w in name.split('_')])
    trial_name = name

    trial = {
        "name": trial_name,
        "c3d_path": c3d_file_path,
        "markers": [],
    }

    if os.path.isfile(c3d_file_path):
        with suppress_output():
            c3d_data = nimble.biomechanics.C3DLoader.loadC3D(c3d_file_path)
        trial["markers"] = c3d_data.markerTimesteps

    return trial

def load_subject(ppid, base_directory):
    def _format(val):
        if pd.isna(val):
            return None
        return str(val).strip().lower().replace(" ", "_").replace("-", "_")

    subject = {
        "ppid": str(ppid),
        "trials": [],
        "calibration": None,
        "sampling_frequency": None,
        "gender": None,
        "age": None,
        "height": None,
        "mass": None,
        "level": None,
        "sport": None,
        "injuries": [],
    }
    # Load metadata
    subject_log_path = os.path.join(base_directory, "Participants Info", "Subject Log.xlsx")
    if os.path.exists(subject_log_path):
        df = pd.read_excel(subject_log_path)
        df.columns = df.columns.str.strip()
        subj_id_col = "Subject ID"
        df[subj_id_col] = df[subj_id_col].astype(str).str.strip().str.lower()
        ppid_str = str(ppid).strip().lower()
        row = df[df[subj_id_col] == ppid_str]
        if not row.empty:
            record = row.iloc[0]
            subject["gender"] = _format(record.get("Gender"))
            subject["age"] = record.get("Age")
            subject["height"] = record.get("Height (cm)")
            subject["mass"] = record.get("Mass (kg)")
            subject["level"] = _format(record.get("Level"))
            subject["sport"] = _format(record.get("Sport"))
            injury_cols = [f"Injury {i}" for i in range(1, 6)]
            subject["injuries"] = [
                _format(record.get(col))
                for col in injury_cols
                if pd.notna(record.get(col)) and str(record.get(col)).strip() != ""
            ]

    # Sampling frequency
    sampling_freq_path = os.path.join(base_directory, "Participants Info", "Sampling Frequency.xlsx")
    if os.path.exists(sampling_freq_path):
        df = pd.read_excel(sampling_freq_path)
        df.columns = df.columns.str.strip()
        df["PPID"] = df["PPID"].astype(str).str.strip().str.lower()
        ppid_str = str(ppid).strip().lower()
        row = df[df["PPID"] == ppid_str]
        if not row.empty and "Sampling Frequency (Hz)" in df.columns:
            subject["sampling_frequency"] = row.iloc[0]["Sampling Frequency (Hz)"]


    # Find all c3d files for this subject
    c3d_folder = os.path.join(base_directory, "Kinematic_Data", str(ppid), "Generated_C3D_files")
    c3d_files = sorted(glob.glob(os.path.join(c3d_folder, "*.c3d")))

    # Create trials from all c3d files
    for c3d_file in c3d_files:
        if os.path.basename(c3d_file).lower() == "calib1.c3d":
            subject["calibration"] = load_trial(c3d_file)
        else:
            trial = load_trial(c3d_file)
            subject["trials"].append(trial)

    return subject

def load_dataset(base_directory, preload=False):
    kinematic_directory = os.path.join(base_directory, "Kinematic_Data")
    ppids = sorted(
        [
            d
            for d in os.listdir(kinematic_directory)
            if os.path.isdir(os.path.join(kinematic_directory, d)) and d.isdigit()
        ],
        key=lambda x: int(x)
    )

    if preload:
        dataset = []
        for ppid in tqdm(ppids, desc="Loading subjects"):
            subject = load_subject(ppid, base_directory)
            dataset.append(subject)
        return {"directory": base_directory, "subject": dataset}
    else:
        return {"directory": base_directory, "ppid": ppids}