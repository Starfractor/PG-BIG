"""
Generate Ground Truth Muscle Activations from Motion Files

This script uses OpenSim to compute muscle activations from .mot files using Static Optimization.
The process involves:
1. Loading the OpenSim model
2. Running Inverse Dynamics to compute joint moments
3. Running Static Optimization to compute muscle activations that produce those moments

Based on OpenCap-processing methodology but simplified for batch processing of existing motion files.

Author: [Your name]
Date: October 2025
"""

import os
import sys
import numpy as np
import opensim
import pandas as pd
from pathlib import Path

# Local minimal replacements for opencap-processing utilities to avoid
# importing their `utils` module which triggers interactive OpenCap login.
def storage_to_dataframe(storage_path):
    """
    Minimal reader for OpenSim .mot/.sto storage files.
    Returns a pandas DataFrame with a 'time' column and the data columns.
    This is intentionally lightweight and supports the standard OpenSim
    storage text format used in this repo.
    """
    storage_path = str(storage_path)
    with open(storage_path, 'r') as f:
        lines = f.readlines()

    # Find the end of the header (line containing 'endheader')
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('endheader'):
            end_idx = i
            break
    if end_idx is None:
        raise ValueError(f"Invalid storage file (no endheader): {storage_path}")

    # The header describing columns is the first non-empty line after endheader
    header_line = None
    header_idx = None
    for j in range(end_idx + 1, len(lines)):
        l = lines[j].strip()
        if l:
            header_line = l
            header_idx = j
            break
    if header_line is None:
        raise ValueError(f"Could not find header line in: {storage_path}")

    # Column names are whitespace separated
    col_names = header_line.split()

    # Data starts on the next line after the header
    data_text = ''.join(lines[header_idx + 1:])
    from io import StringIO
    df = pd.read_csv(StringIO(data_text), sep=r'\s+', names=col_names, comment='#')

    # Normalize time column name
    if 'time' not in df.columns and 'Time' in df.columns:
        df = df.rename(columns={'Time': 'time'})

    return df


def numpy_to_storage(array, times, column_names, out_path):
    """
    Minimal writer to save numpy array with a time vector to a CSV-like file.
    Not a full .sto writer but sufficient for downstream processing in this project.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(array, columns=column_names)
    df.insert(0, 'time', times)
    df.to_csv(out_path, index=False)
    return out_path


class MuscleActivationGenerator:
    """
    Class to generate muscle activations from motion files using OpenSim Static Optimization
    """
    
    def __init__(self, model_path, kinematics_dir, output_dir):
        """
        Initialize the generator
        
        Parameters:
        -----------
        model_path : str
            Path to the scaled OpenSim model (.osim file)
        kinematics_dir : str
            Directory containing the motion (.mot) files
        output_dir : str
            Directory to save the results
        """
        self.model_path = model_path
        self.kinematics_dir = Path(kinematics_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.id_dir = self.output_dir / 'InverseDynamics'
        self.so_dir = self.output_dir / 'StaticOptimization'
        self.activations_dir = self.output_dir / 'Activations'
        
        for dir_path in [self.id_dir, self.so_dir, self.activations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model: {model_path}")
        opensim.Logger.setLevelString('error')
        self.model = opensim.Model(model_path)
        self.model.initSystem()
        
        # Get muscle names
        self.muscle_names = []
        for i in range(self.model.getMuscles().getSize()):
            self.muscle_names.append(self.model.getMuscles().get(i).getName())
        
        print(f"Model loaded successfully with {len(self.muscle_names)} muscles")
    
    
    def run_inverse_dynamics(self, motion_file):
        """
        Run Inverse Dynamics analysis
        
        Parameters:
        -----------
        motion_file : str or Path
            Path to the motion file (.mot)
            
        Returns:
        --------
        id_output_file : Path
            Path to the inverse dynamics results file
        """
        motion_file = Path(motion_file)
        trial_name = motion_file.stem
        
        print(f"  Running Inverse Dynamics for {trial_name}...")
        
        # Get time range from motion file
        motion_data = storage_to_dataframe(str(motion_file))
        time_range = [motion_data['time'].iloc[0], motion_data['time'].iloc[-1]]
        
        # Setup Inverse Dynamics tool
        id_tool = opensim.InverseDynamicsTool()
        id_tool.setModel(self.model)
        id_tool.setName(trial_name)
        
        # Set coordinates file
        id_tool.setCoordinatesFileName(str(motion_file))
        
        # Set time range
        id_tool.setStartTime(time_range[0])
        id_tool.setEndTime(time_range[1])
        
        # Set output
        id_output_file = self.id_dir / f"{trial_name}_id.sto"
        id_tool.setResultsDir(str(self.id_dir))
        id_tool.setOutputGenForceFileName(f"{trial_name}_id.sto")
        
        # Low pass filter settings (optional, set to -1 to disable)
        id_tool.setLowpassCutoffFrequency(-1)
        
        # Run the tool
        try:
            id_tool.run()
            print(f"    ✓ Inverse Dynamics completed")
            return id_output_file
        except Exception as e:
            print(f"    ✗ Error running Inverse Dynamics: {e}")
            return None
    
    
    def run_static_optimization(self, motion_file, id_file):
        """
        Run Static Optimization to compute muscle activations
        
        Parameters:
        -----------
        motion_file : str or Path
            Path to the motion file (.mot)
        id_file : str or Path
            Path to the inverse dynamics results file
            
        Returns:
        --------
        activation_file : Path
            Path to the muscle activations file
        """
        motion_file = Path(motion_file)
        trial_name = motion_file.stem
        
        print(f"  Running Static Optimization for {trial_name}...")
        
        # Get time range from motion file
        motion_data = storage_to_dataframe(str(motion_file))
        time_range = [motion_data['time'].iloc[0], motion_data['time'].iloc[-1]]
        
        # Create a fresh model instance for this analysis
        model = opensim.Model(self.model_path)
        state = model.initSystem()
        
        # Setup Static Optimization tool
        so_tool = opensim.StaticOptimization(model)
        so_tool.setName(trial_name)
        
        # Configure Static Optimization parameters
        so_tool.setActivationExponent(2)
        so_tool.setUseMusclePhysiology(True)
        so_tool.setConvergenceCriterion(1e-4)
        so_tool.setMaxIterations(100)
        
        # Add to model's analysis set
        model.addAnalysis(so_tool)
        
        # Create analysis tool to run static optimization
        analyze_tool = opensim.AnalyzeTool(model)
        analyze_tool.setName(trial_name)
        analyze_tool.setModel(model)
        
        # Set coordinates file
        analyze_tool.setCoordinatesFileName(str(motion_file))
        
        # CRITICAL: Set states file to the same coordinates file
        # This tells AnalyzeTool to use the coordinates as the state trajectory
        # OpenSim will compute muscle states from the joint coordinates
        analyze_tool.setStatesFileName(str(motion_file))
        
        # Set time range
        analyze_tool.setInitialTime(time_range[0])
        analyze_tool.setFinalTime(time_range[1])
        
        # Set results directory
        analyze_tool.setResultsDir(str(self.so_dir))
        
        # Low pass filter settings
        analyze_tool.setLowpassCutoffFrequency(-1)
        
        # Run the tool
        try:
            analyze_tool.run()
            
            # The activation file is typically named: {trial_name}_StaticOptimization_activation.sto
            activation_file = self.so_dir / f"{trial_name}_StaticOptimization_activation.sto"
            
            if activation_file.exists():
                print(f"    ✓ Static Optimization completed")
                return activation_file
            else:
                print(f"    ✗ Activation file not found at expected location: {activation_file}")
                # Try alternative naming
                alt_activation_file = self.so_dir / f"{trial_name}_Actuation_activation.sto"
                if alt_activation_file.exists():
                    print(f"    ✓ Found activation file at: {alt_activation_file}")
                    return alt_activation_file
                return None
                
        except Exception as e:
            print(f"    ✗ Error running Static Optimization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def process_activations(self, activation_file, trial_name):
        """
        Load and process activation results
        
        Parameters:
        -----------
        activation_file : str or Path
            Path to the activation file
        trial_name : str
            Name of the trial
            
        Returns:
        --------
        activations_df : pd.DataFrame
            DataFrame containing muscle activations
        """
        print(f"  Processing activations for {trial_name}...")
        
        try:
            # Load activations
            activations_df = storage_to_dataframe(str(activation_file))
            
            # Save as CSV for easier access
            csv_file = self.activations_dir / f"{trial_name}_activations.csv"
            activations_df.to_csv(csv_file, index=False)
            print(f"    ✓ Saved activations to CSV: {csv_file}")
            
            # Also save as numpy array
            npy_file = self.activations_dir / f"{trial_name}_activations.npy"
            activation_array = activations_df.drop(columns=['time']).values
            np.save(npy_file, activation_array)
            print(f"    ✓ Saved activations to numpy: {npy_file}")
            
            return activations_df
            
        except Exception as e:
            print(f"    ✗ Error processing activations: {e}")
            return None
    
    
    def process_single_trial(self, motion_file):
        """
        Process a single motion file to generate muscle activations
        
        Parameters:
        -----------
        motion_file : str or Path
            Path to the motion file
            
        Returns:
        --------
        success : bool
            Whether processing was successful
        """
        motion_file = Path(motion_file)
        trial_name = motion_file.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {trial_name}")
        print(f"{'='*60}")
        
        # Step 1: Run Inverse Dynamics
        id_file = self.run_inverse_dynamics(motion_file)
        if id_file is None:
            return False
        
        # Step 2: Run Static Optimization
        activation_file = self.run_static_optimization(motion_file, id_file)
        if activation_file is None:
            return False
        
        # Step 3: Process and save activations
        activations_df = self.process_activations(activation_file, trial_name)
        if activations_df is None:
            return False
        
        print(f"\n✓ Successfully processed {trial_name}")
        print(f"  - Activation shape: {activations_df.shape}")
        print(f"  - Number of muscles: {len(activations_df.columns) - 1}")  # -1 for time column
        
        return True
    
    
    def process_all_trials(self, pattern="*.mot"):
        """
        Process all motion files in the kinematics directory
        
        Parameters:
        -----------
        pattern : str
            File pattern to match (default: "*.mot")
            
        Returns:
        --------
        results : dict
            Dictionary with trial names as keys and success status as values
        """
        motion_files = sorted(self.kinematics_dir.glob(pattern))
        
        if len(motion_files) == 0:
            print(f"No motion files found in {self.kinematics_dir} with pattern {pattern}")
            return {}
        
        print(f"\nFound {len(motion_files)} motion files to process")
        
        results = {}
        for motion_file in motion_files:
            trial_name = motion_file.stem
            success = self.process_single_trial(motion_file)
            results[trial_name] = success
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        successful = sum(results.values())
        print(f"Successfully processed: {successful}/{len(results)} trials")
        
        if successful < len(results):
            print("\nFailed trials:")
            for trial, success in results.items():
                if not success:
                    print(f"  - {trial}")
        
        return results


def main():
    """
    Main function to generate muscle activations for all trials
    """
    
    # Configuration
    model_path = "/home/mnt/code/opencap/Data/OpenSimData/Kinematics/Rajagopal2015.osim"
    kinematics_dir = "/home/mnt/code/opencap/Data/OpenSimData/Kinematics"
    output_dir = "/home/mnt/code/PG-BIG/muscle_activations_output"
    
    print("="*60)
    print("Muscle Activation Generation Pipeline")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Kinematics: {kinematics_dir}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("\nPlease provide the correct path to your scaled OpenSim model (.osim file)")
        return
    
    # Check if kinematics directory exists
    if not os.path.exists(kinematics_dir):
        print(f"ERROR: Kinematics directory not found: {kinematics_dir}")
        return
    
    # Create generator
    generator = MuscleActivationGenerator(
        model_path=model_path,
        kinematics_dir=kinematics_dir,
        output_dir=output_dir
    )
    
    # Process specific subjects/trials or all
    # Option 1: Process specific trials
    # specific_trials = [
    #     "subject_38_action_drop_jump.mot",
    #     "subject_39_action_drop_jump.mot"
    # ]
    # for trial in specific_trials:
    #     trial_path = Path(kinematics_dir) / trial
    #     if trial_path.exists():
    #         generator.process_single_trial(trial_path)
    
    # Option 2: Process all trials matching a pattern
    # Process all subject 38 and 39 trials
    #results = generator.process_all_trials(pattern="subject_3[89]_*.mot")
    
    # Option 3: Process ALL trials
    results = generator.process_all_trials(pattern="*.mot")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
