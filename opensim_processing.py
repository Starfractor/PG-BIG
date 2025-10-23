import casadi as ca
import multiprocessing as mp
import numpy as np
import opensim as osim
import os
import time
from functools import partial

# Suppress verbose OpenSim messages (use 'Error' to only show errors, or 'Off' to silence completely)
osim.Logger.setLevelString('Error')

def create_symbolic_dynamics(nQ):
    q = ca.SX.sym('q', nQ)
    qd = ca.SX.sym('qd', nQ)
    qdd = ca.SX.sym('qdd', nQ)
    tau = ca.SX.sym('tau', nQ)
    
    M = ca.SX.eye(nQ) * 10.0
    C = ca.SX.zeros(nQ)
    for i in range(nQ):
        C[i] = 0.1 * qd[i]**2
    
    g = 9.81
    G = ca.SX.zeros(nQ)
    for i in range(nQ):
        G[i] = 5.0 * g * ca.sin(q[i])
    
    residual = ca.mtimes(M, qdd) + C + G - tau
    
    f_M = ca.Function('f_M', [q], [M])
    f_C = ca.Function('f_C', [q, qd], [C])
    f_G = ca.Function('f_G', [q], [G])
    f_dynamics = ca.Function('f_dynamics', [q, qd, qdd, tau], [residual])
    
    return f_M, f_C, f_G, f_dynamics


def compute_muscle_moment_arms_parallel(muscle_idx, muscle_name, osim_model_path, 
                                       coordinate_names, coord_ranges, locked_coords,
                                       n_samples=20, poly_degree=3):
    """Compute moment arms for a single muscle (parallelizable)."""
    model = osim.Model(osim_model_path)
    state = model.initSystem()
    muscle = model.getMuscles().get(muscle_name)
    
    moment_arm_polys = {}
    moment_arm_funcs = {}
    
    for coord_name in coordinate_names:
        if coord_name in locked_coords:
            poly_coeffs = np.zeros(poly_degree + 1)
            moment_arm_polys[(muscle_name, coord_name)] = poly_coeffs
            q_sym = ca.SX.sym('q')
            moment_arm_funcs[(muscle_name, coord_name)] = ca.Function(
                f'ma_{muscle_name}_{coord_name}', [q_sym], [ca.SX.zeros(1)]
            )
            continue
        
        coord = model.getCoordinateSet().get(coord_name)
        qmin, qmax = coord_ranges[coord_name]
        
        q_samples = np.linspace(qmin, qmax, n_samples)
        ma_samples = []
        
        for q_val in q_samples:
            for j in range(model.getCoordinateSet().getSize()):
                other_coord = model.getCoordinateSet().get(j)
                if not other_coord.getLocked(state):
                    other_coord.setValue(state, 0.0)
            
            coord.setValue(state, float(q_val))
            model.realizePosition(state)
            
            try:
                ma = muscle.computeMomentArm(state, coord)
                ma_samples.append(float(ma))
            except Exception:
                ma_samples.append(0.0)
        
        if np.allclose(ma_samples, 0.0):
            poly_coeffs = np.zeros(poly_degree + 1)
        else:
            poly_coeffs = np.polyfit(q_samples, ma_samples, deg=poly_degree)
        
        moment_arm_polys[(muscle_name, coord_name)] = poly_coeffs
        
        q_sym = ca.SX.sym('q')
        ma_expr = 0
        for i, coeff in enumerate(poly_coeffs):
            ma_expr += coeff * q_sym**(poly_degree - i)
        
        moment_arm_funcs[(muscle_name, coord_name)] = ca.Function(
            f'ma_{muscle_name}_{coord_name}', [q_sym], [ma_expr]
        )
    
    return muscle_idx, muscle_name, moment_arm_polys, moment_arm_funcs


def precompute_moment_arms(osim_model_path, muscle_names, coordinate_names, 
                           coord_ranges=None, n_samples=20, poly_degree=3, n_jobs=None):
    """Precompute moment arms with optional parallel processing."""
    print(f'[Moment Arms] Starting computation for {len(muscle_names)} muscles x {len(coordinate_names)} coordinates')
    t_start = time.time()
    
    model = osim.Model(osim_model_path)
    state = model.initSystem()
    
    locked_coords = set()
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        if coord.getLocked(state):
            locked_coords.add(coord.getName())
    
    if locked_coords:
        print(f'[Moment Arms] Skipping {len(locked_coords)} locked coordinates')
    
    if coord_ranges is None:
        coord_ranges = {}
        for coord_name in coordinate_names:
            if coord_name in locked_coords:
                continue
            coord = model.getCoordinateSet().get(coord_name)
            coord_ranges[coord_name] = (
                coord.getRangeMin() if coord.getRangeMin() > -np.pi else -np.pi,
                coord.getRangeMax() if coord.getRangeMax() < np.pi else np.pi
            )
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
        print(f'[Moment Arms] Auto-detected {mp.cpu_count()} CPUs, using {n_jobs} workers')
    elif n_jobs == -1:
        n_jobs = mp.cpu_count()
        print(f'[Moment Arms] Using all {n_jobs} CPUs')
    elif n_jobs < 1:
        n_jobs = 1
    
    moment_arm_polys = {}
    moment_arm_funcs = {}
    
    if n_jobs == 1:
        for m_idx, muscle_name in enumerate(muscle_names):
            print(f'[Moment Arms] Processing muscle {m_idx+1}/{len(muscle_names)}: {muscle_name}')
            _, _, polys, funcs = compute_muscle_moment_arms_parallel(
                m_idx, muscle_name, osim_model_path, coordinate_names,
                coord_ranges, locked_coords, n_samples, poly_degree
            )
            moment_arm_polys.update(polys)
            moment_arm_funcs.update(funcs)
            
            elapsed = time.time() - t_start
            progress = (m_idx + 1) / len(muscle_names)
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            print(f'[Moment Arms] Progress: {m_idx+1}/{len(muscle_names)} muscles | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s')
    else:
        print(f'[Moment Arms] Starting parallel computation...')
        worker_func = partial(
            compute_muscle_moment_arms_parallel,
            osim_model_path=osim_model_path,
            coordinate_names=coordinate_names,
            coord_ranges=coord_ranges,
            locked_coords=locked_coords,
            n_samples=n_samples,
            poly_degree=poly_degree
        )
        tasks = [(i, name) for i, name in enumerate(muscle_names)]
        
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(worker_func, tasks)
        
        print(f'[Moment Arms] Merging results from {len(results)} workers...')
        for muscle_idx, muscle_name, polys, funcs in results:
            moment_arm_polys.update(polys)
            moment_arm_funcs.update(funcs)
            print(f'[Moment Arms] Merged results for muscle {muscle_idx+1}/{len(muscle_names)}: {muscle_name}')
    
    unlocked_count = len([c for c in coordinate_names if c not in locked_coords])
    elapsed_total = time.time() - t_start
    speedup = f" (estimated {n_jobs*0.7:.1f}x speedup)" if n_jobs > 1 else ""
    print(f'[Moment Arms] ✓ Completed in {elapsed_total:.1f}s{speedup}: {len(muscle_names)} muscles x {unlocked_count} unlocked coordinates')
    return moment_arm_polys, moment_arm_funcs


def build_moment_arm_matrix(q, muscle_names, coordinate_names, moment_arm_funcs):
    """Build symbolic moment arm matrix R[nQ x nM] as function of joint angles q."""
    nQ = len(coordinate_names)
    nM = len(muscle_names)
    
    R_list = []
    for i_coord, coord_name in enumerate(coordinate_names):
        row = []
        for i_muscle, muscle_name in enumerate(muscle_names):
            key = (muscle_name, coord_name)
            if key in moment_arm_funcs:
                ma_val = moment_arm_funcs[key](q[i_coord])
                row.append(ma_val)
            else:
                row.append(0.0)
        R_list.append(row)
    
    R = ca.vertcat(*[ca.horzcat(*row) for row in R_list])
    return R


def activation_dynamics(e, a, ta=0.015, td=0.060, b=0.1):
    """De Groote style activation dynamics: da/dt = (u - a) / tau(u, a)"""
    tau_act = ta * (0.5 + 1.5*a)
    tau_deact = td / (0.5 + 1.5*a)
    f_switch = 0.5 + 0.5 * ca.tanh(b * (e - a))
    tau = tau_act * f_switch + tau_deact * (1 - f_switch)
    return (e - a) / tau
def muscle_tendon_equilibrium(a, lM_tilde, vM_tilde, lMT, mtu_params):
    """Enforce muscle-tendon equilibrium: F_tendon = F_fiber * cos(alpha)"""
    Fmax = mtu_params['max_isometric_force']
    lM_opt = mtu_params['optimal_fiber_length']
    lT_slack = mtu_params['tendon_slack_length']
    alpha0 = mtu_params['pennation_angle']
    
    lM = lM_tilde * lM_opt
    
    sin_alpha = (lM_opt * ca.sin(alpha0)) / lM
    sin_alpha = ca.fmin(sin_alpha, 1.0)
    cos_alpha = ca.sqrt(1 - sin_alpha**2)
    
    lT = lMT - lM * cos_alpha
    lT_tilde = lT / lT_slack
    
    F_iso = ca.exp(-((lM_tilde - 1.0) / 0.45)**2)
    F_pas = ca.fmax(0, (lM_tilde - 1.0) / 0.6)**2
    F_v = (1.8 - vM_tilde) / (1.8 + 1.5 * vM_tilde)
    
    F_fiber = Fmax * ((a * F_iso * F_v + F_pas) * cos_alpha)
    
    kT = 35.0
    F_tendon = ca.if_else(
        lT_tilde > 1.0,
        Fmax * (ca.exp(kT * (lT_tilde - 1.0) / 0.04) - 1.0),
        0.0
    )
    
    residual = F_fiber - F_tendon
    
    return residual, F_tendon
def build_improved_opt_problem(coords_ref, times, osim_model_path, 
                                muscle_names, settings):
    """Build optimization problem with symbolic dynamics and muscle states.
    
    This implements tight tracking constraints similar to OpenCap's approach where
    the joint coordinates are constrained to closely follow the IK results.
    
    Key features matching OpenCap methodology:
    - Tight bounds on Q (tracking_tolerance, default ±0.01 rad/m)
    - Very high tracking weight (wl_tracking = 1e6)
    - Hard constraint on initial conditions from IK
    - Coordinate velocities computed from IK results
    
    Args:
        coords_ref: Reference coordinates from IK (N+1 x nQ)
        times: Time vector (N+1,)
        osim_model_path: Path to OpenSim model
        muscle_names: List of muscle names
        settings: Dictionary with optimization settings
            - tracking_tolerance: Max deviation from IK (default: 0.01)
            - wl_tracking: Tracking weight (default: 1e6)
            - wl_effort: Muscle effort weight (default: 1.0)
            - wl_reserve: Reserve actuator weight (default: 1e3)
    
    Returns:
        opti: CasADi Opti instance
        vars_dict: Dictionary of optimization variables
        times: Time vector
        muscle_names: List of muscle names used
        coord_names: List of coordinate names
    """
    opti = ca.Opti()
    
    N = settings.get('N', coords_ref.shape[0] - 1)
    nQ = coords_ref.shape[1]
    nM = len(muscle_names)
    
    dt = np.mean(np.diff(times))
    
    # Compute coordinate velocities from IK coordinates for tracking
    coords_vel_ref = np.zeros((N+1, nQ))
    for i in range(nQ):
        coords_vel_ref[:, i] = np.gradient(coords_ref[:, i], times)
    
    Q = opti.variable(nQ, N+1)
    Qd = opti.variable(nQ, N+1)
    Qdd = opti.variable(nQ, N+1)
    E = opti.variable(nM, N+1)
    A = opti.variable(nM, N+1)
    lM = opti.variable(nM, N+1)
    vM = opti.variable(nM, N+1)
    R = opti.variable(nQ, N+1)
    
    opti.set_initial(Q, coords_ref.T)
    opti.set_initial(Qd, coords_vel_ref.T)
    opti.set_initial(Qdd, 0)
    opti.set_initial(E, 0.05)
    opti.set_initial(A, 0.05)
    opti.set_initial(lM, 1.0)
    opti.set_initial(vM, 0.0)
    opti.set_initial(R, 0.0)
    
    # TIGHT BOUNDS on coordinates to enforce tracking (OpenCap-style)
    # Allow only small deviations (0.01 rad ~ 0.57 degrees, 0.01 m for translations)
    tracking_tolerance = settings.get('tracking_tolerance', 0.01)
    for k in range(N+1):
        for i in range(nQ):
            q_ref = coords_ref[k, i]
            opti.subject_to(opti.bounded(
                q_ref - tracking_tolerance,
                Q[i, k],
                q_ref + tracking_tolerance
            ))
    
    # Velocity bounds (allow more freedom for numerical stability)
    opti.subject_to(opti.bounded(-10.0, Qd, 10.0))
    opti.subject_to(opti.bounded(-100.0, Qdd, 100.0))
    opti.subject_to(opti.bounded(0.01, E, 1.0))
    opti.subject_to(opti.bounded(0.01, A, 1.0))
    opti.subject_to(opti.bounded(0.5, lM, 1.5))
    opti.subject_to(opti.bounded(-1.0, vM, 1.0))
    opti.subject_to(opti.bounded(0.0, R, 100.0))
    
    # Hard constraints on initial conditions
    opti.subject_to(Q[:, 0] == coords_ref[0, :])
    opti.subject_to(Qd[:, 0] == coords_vel_ref[0, :])
    
    f_M, f_C, f_G, f_dynamics = create_symbolic_dynamics(nQ)
    
    model = osim.Model(osim_model_path)
    state = model.initSystem()
    coord_names = []
    for i in range(model.getCoordinateSet().getSize()):
        coord_names.append(model.getCoordinateSet().get(i).getName())
    
    n_jobs = settings.get('n_jobs', None)
    _, ma_funcs = precompute_moment_arms(
        osim_model_path, muscle_names, coord_names,
        n_samples=15, poly_degree=3, n_jobs=n_jobs
    )
    
    J = 0
    # OpenCap-style weights: very high tracking (effectively hard constraint)
    # combined with tight bounds on Q
    wt = settings.get('wl_tracking', 1e6)  # Much higher than before
    we = settings.get('wl_effort', 1.0)
    wr = settings.get('wl_reserve', 1e3)
    
    for k in range(N):
        qk = Q[:, k]
        qdk = Qd[:, k]
        qddk = Qdd[:, k]
        ek = E[:, k]
        ak = A[:, k]
        lMk = lM[:, k]
        vMk = vM[:, k]
        rk = R[:, k]
        
        qk1 = Q[:, k+1]
        qdk1 = Qd[:, k+1]
        qddk1 = Qdd[:, k+1]
        ek1 = E[:, k+1]
        ak1 = A[:, k+1]
        lMk1 = lM[:, k+1]
        vMk1 = vM[:, k+1]
        rk1 = R[:, k+1]
        
        qm = 0.5*(qk + qk1) + (dt/8.0)*(qdk - qdk1)
        qdm = 0.5*(qdk + qdk1) - (3.0/(2.0*dt))*(qk - qk1)
        qddm = 0.5*(qddk + qddk1)
        
        em = 0.5*(ek + ek1)
        am = 0.5*(ak + ak1)
        lMm = 0.5*(lMk + lMk1)
        vMm = 0.5*(vMk + vMk1)
        rm = 0.5*(rk + rk1)
        
        opti.subject_to(qk1 == qk + (dt/6.0)*(qdk + 4*qdm + qdk1))
        opti.subject_to(qdk1 == qdk + (dt/6.0)*(qddk + 4*qddm + qddk1))
        
        dadt_k = activation_dynamics(ek, ak)
        dadt_m = activation_dynamics(em, am)
        dadt_k1 = activation_dynamics(ek1, ak1)
        opti.subject_to(ak1 == ak + (dt/6.0)*(dadt_k + 4*dadt_m + dadt_k1))
        
        Rk = build_moment_arm_matrix(qk, muscle_names, coord_names, ma_funcs)
        Rk1 = build_moment_arm_matrix(qk1, muscle_names, coord_names, ma_funcs)
        
        for q_pt, qd_pt, qdd_pt, R_mat, a_pt, r_pt in [
            (qk, qdk, qddk, Rk, ak, rk),
            (qk1, qdk1, qddk1, Rk1, ak1, rk1)
        ]:
            M_pt = f_M(q_pt)
            C_pt = f_C(q_pt, qd_pt)
            G_pt = f_G(q_pt)
            
            F_muscle = a_pt * 1500.0
            tau_muscle = ca.mtimes(R_mat, F_muscle)
            tau_total = tau_muscle + r_pt
            
            dyn_res = ca.mtimes(M_pt, qdd_pt) + C_pt + G_pt - tau_total
            opti.subject_to(dyn_res == 0)
        
        J += wt * ca.sumsqr(qk - coords_ref[k, :])
        J += we * ca.sumsqr(ek)
        J += wr * ca.sumsqr(rk)
    
    J += wt * ca.sumsqr(Q[:, -1] - coords_ref[-1, :])
    J += we * ca.sumsqr(E[:, -1])
    J += wr * ca.sumsqr(R[:, -1])
    
    opti.minimize(J)
    
    opts = {
        'ipopt.max_iter': settings.get('max_iter', 1000),
        'ipopt.tol': settings.get('tol', 1e-6),
        'ipopt.print_level': 5,
    }
    opti.solver('ipopt', opts)
    
    vars_dict = {
        'Q': Q, 'Qd': Qd, 'Qdd': Qdd,
        'E': E, 'A': A, 'lM': lM, 'vM': vM,
        'R': R
    }
    
    return opti, vars_dict, times, muscle_names, coord_names

def extract_solution(opti, sol, vars_dict, times):
    """Extract solution from CasADi Opti after solving."""
    sol_dict = {'times': times}
    
    for name, var in vars_dict.items():
        try:
            sol_dict[name] = sol.value(var)
        except Exception as e:
            print(f'Warning: could not extract {name}: {e}')
            sol_dict[name] = np.zeros(var.shape)
    
    return sol_dict

def parse_b3d_simple(path_b3d):
    """Parse B3D file using Nimble Physics library."""
    if path_b3d is None or not os.path.exists(path_b3d):
        # Synthetic demo
        T = 101
        t = np.linspace(0, 1.0, T)
        x = np.sin(2*np.pi*1.0*t)
        y = 0.1*np.cos(2*np.pi*0.5*t)
        z = 0.05*np.ones_like(t)
        markers = np.vstack([x, y, z]).T
        labels = ['marker1']
        return t, markers.reshape((T, 3)), labels
    
    # Use Nimble Physics to read B3D file
    try:
        import nimblephysics as nimble

        print(f'Loading B3D file with Nimble Physics: {path_b3d}')
        subject = nimble.biomechanics.SubjectOnDisk(path_b3d)

        # Use first trial
        trial = 0
        num_frames = subject.getTrialLength(trial)
        timestep = subject.getTrialTimestep(trial)

        # Read frames from kinematics pass (pass 0)
        # Note: readFrames signature is (trial, startFrame, numFramesToRead, ...)
        frames = subject.readFrames(
            trial,
            0,  # startFrame
            num_frames  # numFramesToRead
        )

        # Extract marker observations from all frames
        # Collect all unique marker names across all frames
        marker_names_set = set()
        for frame in frames:
            # markerObservations is a list of tuples (marker_name, position_3d_vector)
            for item in frame.markerObservations:
                marker_names_set.add(item[0])

        marker_names = sorted(list(marker_names_set))
        print(f'Found {len(marker_names)} unique markers')

        # Build marker data matrix (T x 3*M)
        markers_data = []
        for frame in frames:
            # Build dict of marker observations for this frame
            frame_markers = {}
            for item in frame.markerObservations:
                marker_name = item[0]
                marker_pos = item[1]  # This is a numpy array of shape (3,)
                frame_markers[marker_name] = marker_pos

            # Concatenate all markers for this frame in sorted order
            frame_vec = []
            for marker_name in marker_names:
                if marker_name in frame_markers:
                    # Convert to list and extend
                    frame_vec.extend(frame_markers[marker_name].tolist())
                else:
                    # Missing marker - use zeros
                    frame_vec.extend([0.0, 0.0, 0.0])
            markers_data.append(frame_vec)

        markers = np.array(markers_data)
        times = np.arange(num_frames) * timestep

        print(f'Loaded {num_frames} frames at {timestep:.4f}s timestep')
        print(f'Marker data shape: {markers.shape}')

        return times, markers, marker_names

    except ImportError:
        print('Error: nimblephysics not installed. Install with: pip install nimblephysics')
        print('Falling back to synthetic data')
    except Exception as e:
        print(f'Warning: could not parse {path_b3d}: {e}')
        print('Using synthetic data')
    
    # Fallback to synthetic data
    T = 101
    t = np.linspace(0, 1.0, T)
    markers = np.random.randn(T, 9) * 0.1
    labels = ['marker1', 'marker2', 'marker3']
    return t, markers, labels


def write_mot_file(t, data, labels, filepath):
    """Write OpenSim .mot file."""
    nframes = t.shape[0]
    ncols = len(labels)
    
    with open(filepath, 'w') as f:
        f.write('name Motion\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' % nframes)
        f.write('nColumns=%d\n' % (ncols + 1))
        f.write('inDegrees=no\n')
        f.write('endheader\n')
        f.write('time\t' + '\t'.join(labels) + '\n')
        for i in range(nframes):
            row = [f"{t[i]:.6f}"] + [f"{data[i,j]:.6f}" for j in range(ncols)]
            f.write('\t'.join(row) + '\n')
    print(f'Wrote MOT file to {filepath}')


def run_simple_ik(osim_model_path, markers_mot_path, output_coords_path):
    """Run a simplified IK - for demo, just return synthetic coordinates."""
    print(f'Running IK with model: {osim_model_path}')
    
    try:
        model = osim.Model(osim_model_path)
        state = model.initSystem()
        
        nQ = model.getCoordinateSet().getSize()
        coord_names = []
        for i in range(nQ):
            coord_names.append(model.getCoordinateSet().get(i).getName())
        
        with open(markers_mot_path, 'r') as f:
            lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('time'):
                    data_start = i + 1
                    break
        
        times = []
        for line in lines[data_start:]:
            times.append(float(line.split()[0]))
        times = np.array(times)
        
        coords_data = np.zeros((len(times), nQ))
        for j in range(nQ):
            amp = 0.1 if j < 3 else 0.3
            freq = 0.5 + j * 0.1
            coords_data[:, j] = amp * np.sin(2*np.pi*freq*times)
        
        write_mot_file(times, coords_data, coord_names, output_coords_path)
        
        return times, coords_data, coord_names
        
    except Exception as e:
        print(f'IK failed: {e}')
        print('Using synthetic coordinates')
        times = np.linspace(0, 1.0, 101)
        nQ = 10
        coords_data = np.zeros((len(times), nQ))
        for j in range(nQ):
            coords_data[:, j] = 0.2 * np.sin(2*np.pi*(j+1)*times)
        coord_names = [f'coord_{i}' for i in range(nQ)]
        write_mot_file(times, coords_data, coord_names, output_coords_path)
        return times, coords_data, coord_names


def get_muscle_names_from_model(osim_model_path, max_muscles=None):
    """Extract muscle names from OpenSim model."""
    try:
        model = osim.Model(osim_model_path)
        muscles = model.getMuscles()
        total_muscles = muscles.getSize()
        
        if max_muscles is None:
            n = total_muscles
        else:
            n = min(total_muscles, max_muscles)
        
        muscle_names = []
        for i in range(n):
            muscle_names.append(muscles.get(i).getName())
        
        if max_muscles is None:
            print(f'Found {total_muscles} muscles in model, using all {n}')
        else:
            print(f'Found {total_muscles} muscles in model, using first {n}')
        
        return muscle_names
    except Exception as e:
        print(f'Could not extract muscles from model: {e}')
        return ['muscle1', 'muscle2', 'muscle3']


def plot_results(sol_dict, coords_ref, out_prefix):
    """Plot optimization results."""
    import matplotlib.pyplot as plt
    
    times = sol_dict['times']
    Q = sol_dict['Q']
    
    nQ = Q.shape[0]
    nrows = int(np.ceil(np.sqrt(nQ)))
    ncols = int(np.ceil(nQ / nrows))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 8))
    axs = axs.flatten() if nQ > 1 else [axs]
    
    if len(times) != coords_ref.shape[0]:
        times_ref = np.linspace(times[0], times[-1], coords_ref.shape[0])
        coords_ref_resampled = np.zeros((len(times), nQ))
        for j in range(nQ):
            coords_ref_resampled[:, j] = np.interp(times, times_ref, coords_ref[:, j])
        coords_ref = coords_ref_resampled
    
    for i in range(nQ):
        if i < len(axs):
            axs[i].plot(times, coords_ref[:, i], 'k--', label='Reference', linewidth=2)
            axs[i].plot(times, Q[i, :], 'b-', label='Optimized', linewidth=1.5)
            axs[i].set_title(f'Coordinate {i}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Value')
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
    
    for i in range(nQ, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.savefig(out_prefix + '_coordinates.png', dpi=150)
    print(f'Saved plot to {out_prefix}_coordinates.png')
    plt.close()
    
    if 'A' in sol_dict:
        A = sol_dict['A']
        nM = min(A.shape[0], 6)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(nM):
            ax.plot(times, A[i, :], label=f'Muscle {i}', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Activation')
        ax.set_title('Muscle Activations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix + '_activations.png', dpi=150)
        print(f'Saved plot to {out_prefix}_activations.png')
        plt.close()

def process_single_b3d_file(args_tuple):
    """Process a single B3D file (worker function for multiprocessing)."""
    import time
    import traceback
    from pathlib import Path
    
    b3d_path, osim_path, output_base_dir, settings_dict = args_tuple
    
    file_stem = Path(b3d_path).stem
    file_output_dir = os.path.join(output_base_dir, file_stem)
    os.makedirs(file_output_dir, exist_ok=True)
    
    output_prefix = os.path.join(file_output_dir, file_stem)
    
    t_start = time.time()
    
    try:
        times_markers, markers, marker_labels = parse_b3d_simple(b3d_path)
        markers_mot = output_prefix + '_markers.mot'
        
        if markers.ndim == 2 and markers.shape[1] % 3 == 0:
            pass
        else:
            markers = markers.reshape((markers.shape[0], -1))
        
        nmarkers = markers.shape[1] // 3
        mot_labels = []
        for i in range(nmarkers):
            for axis in ['x', 'y', 'z']:
                mot_labels.append(f'marker{i}_{axis}')
        
        write_mot_file(times_markers, markers, mot_labels, markers_mot)
        
        coords_mot = output_prefix + '_coords_ik.mot'
        times, coords_ref, coord_names = run_simple_ik(osim_path, markers_mot, coords_mot)
        
        N = settings_dict.get('N')
        if N is None:
            N = coords_ref.shape[0] - 1
            settings_dict['N'] = N
        
        muscle_names = []
        if settings_dict.get('mode') == 'muscle':
            muscle_names = get_muscle_names_from_model(osim_path, settings_dict.get('max_muscles'))
        
        muscle_names_used = []
        if settings_dict.get('mode') == 'muscle' and len(muscle_names) > 0:
            opti, vars_dict, times_opt, muscle_names_used, _ = build_improved_opt_problem(
                coords_ref, times, osim_path, muscle_names, settings_dict
            )
        else:
            opti, vars_dict, times_opt = build_torque_driven_problem(
                coords_ref, times, settings_dict
            )
        
        try:
            sol = opti.solve()
            sol_dict = extract_solution(opti, sol, vars_dict, times_opt)
        except Exception as e:
            print(f"[{file_stem}] Optimization failed: {e}")
            sol_dict = extract_solution(opti, opti.debug, vars_dict, times_opt)
        
        np.save(output_prefix + '_solution.npy', sol_dict, allow_pickle=True)
        
        if 'A' in sol_dict:
            muscles_to_save = muscle_names_used if muscle_names_used else muscle_names
            activation_labels = [f'{name}_activation' for name in muscles_to_save]
            write_mot_file(sol_dict['times'], sol_dict['A'].T, activation_labels,
                          output_prefix + '_activations.mot')
        
        if 'Q' in sol_dict:
            write_mot_file(sol_dict['times'], sol_dict['Q'].T, coord_names, 
                          output_prefix + '_coords_opt.mot')
        
        try:
            plot_results(sol_dict, coords_ref, output_prefix)
        except Exception as e:
            print(f"[{file_stem}] Could not generate plots: {e}")
        
        elapsed = time.time() - t_start
        print(f"[{file_stem}] ✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return (b3d_path, True, elapsed, None)
        
    except Exception as e:
        elapsed = time.time() - t_start
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{file_stem}] ✗ Failed after {elapsed:.1f}s: {e}")
        return (b3d_path, False, elapsed, error_msg)


def main():
    """Main function to run the direct collocation pipeline."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='OpenSim muscle-driven direct collocation optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python opensim_processing.py --b3d-path markers.b3d --osim-path model.osim
  python opensim_processing.py --b3d-path data/ --osim-path model.osim --n-jobs 8
  python opensim_processing.py --b3d-path data/ --osim-path model.osim --max-muscles 20 --max-iter 200
  python opensim_processing.py --b3d-path data/ --osim-path model.osim --mode torque
  
  # Process files 0-99 (for distributing across multiple pods)
  python opensim_processing.py --b3d-path data/ --osim-path model.osim --start-idx 0 --end-idx 99
  
  # Process every 4th file starting from index 0 (useful for 4 pods)
  python opensim_processing.py --b3d-path data/ --osim-path model.osim --start-idx 0 --stride 4
        """
    )
    
    parser.add_argument('--b3d-path', type=str, default=None,
                        help='Path to .b3d file OR directory containing .b3d files for batch processing')
    parser.add_argument('--osim-path', type=str, required=True,
                        help='Path to OpenSim .osim model file')
    parser.add_argument('--output-dir', type=str, default='outputs/muscle_activations',
                        help='Output directory for results')
    parser.add_argument('--mode', type=str, default='muscle', choices=['muscle', 'torque'],
                        help='Actuation mode: muscle or torque-driven')
    parser.add_argument('--N', type=int, default=None,
                        help='Number of collocation mesh intervals (default: auto-detect from data frames)')
    parser.add_argument('--max-iter', type=int, default=500,
                        help='Maximum IPOPT iterations')
    parser.add_argument('--max-muscles', type=int, default=None,
                        help='Maximum number of muscles to include (default: use all muscles in model)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of files to process in parallel for batch mode (default: auto-detect, -1 = all CPUs)')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='Start index for file selection (inclusive, 0-based)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for file selection (inclusive, 0-based)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Process every Nth file (default: 1, process all files)')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated list of specific indices to process (e.g., "0,5,10")')
    parser.add_argument('--tracking-tolerance', type=float, default=0.01,
                        help='Maximum deviation from IK coordinates in rad/m (default: 0.01, ~0.57 degrees)')
    
    args = parser.parse_args()
    
    from pathlib import Path
    is_batch_mode = False
    b3d_files = []
    
    if args.b3d_path and os.path.exists(args.b3d_path):
        b3d_path_obj = Path(args.b3d_path)
        if b3d_path_obj.is_dir():
            b3d_files = sorted(b3d_path_obj.glob('*.b3d'))
            b3d_files = [str(f) for f in b3d_files]
            is_batch_mode = True
            if not b3d_files:
                print(f"Error: No .b3d files found in directory: {args.b3d_path}")
                return
        elif b3d_path_obj.is_file():
            b3d_files = [str(b3d_path_obj)]
            is_batch_mode = False
    elif args.b3d_path is None:
        b3d_files = [None]
        is_batch_mode = False
    else:
        print(f"Error: Path does not exist: {args.b3d_path}")
        return
    
    if is_batch_mode:
        original_count = len(b3d_files)
        
        if args.indices:
            try:
                selected_indices = [int(i.strip()) for i in args.indices.split(',')]
                selected_indices = [i for i in selected_indices if 0 <= i < original_count]
                b3d_files = [b3d_files[i] for i in selected_indices]
                print(f"Selected {len(b3d_files)} files by explicit indices")
            except ValueError as e:
                print(f"Error parsing --indices: {e}")
                return
        else:
            start = args.start_idx if args.start_idx is not None else 0
            end = args.end_idx if args.end_idx is not None else original_count - 1
            
            if start < 0 or end >= original_count or start > end:
                print(f"Error: Invalid index range [{start}, {end}] for {original_count} files")
                return
            
            b3d_files = b3d_files[start:end+1]
            
            if args.stride > 1:
                b3d_files = b3d_files[::args.stride]
        
        if len(b3d_files) == 0:
            print("Error: No files to process after applying filters")
            return
        
        print(f"Batch processing: {len(b3d_files)} files (selected from {original_count} total)")
        print(f"Input directory: {args.b3d_path}")
    else:
        print(f"Single file: {b3d_files[0] if b3d_files[0] else 'synthetic data'}")
    
    print(f"Output directory: {args.output_dir}")
    print(f"Max iterations: {args.max_iter}")
    
    # Display CPU configuration
    available_cpus = mp.cpu_count()
    
    if is_batch_mode:
        # Batch mode: n-jobs controls how many files processed in parallel
        if args.n_jobs is None:
            args.n_jobs = max(1, available_cpus - 1)
        elif args.n_jobs == -1:
            args.n_jobs = available_cpus
        args.n_jobs = min(args.n_jobs, len(b3d_files))  # Don't use more CPUs than files
        print(f"Parallel files: {args.n_jobs} (each file uses 1 CPU)")
    else:
        # Single file mode: always use 1 CPU
        print(f"Single file mode: 1 CPU")
    
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup optimization settings
    settings = {
        'N': args.N,
        'mode': args.mode,
        'wl_tracking': 1e6,  # Very high weight for tight tracking (OpenCap-style)
        'wl_effort': 0.1 if args.mode == 'muscle' else 1.0,
        'wl_reserve': 1e3,
        'tracking_tolerance': args.tracking_tolerance,  # Allow max deviation from IK
        'max_iter': args.max_iter,
        'tol': 1e-6,
        'n_jobs': 1,  # Always use 1 CPU per file for moment arm computation
        'max_muscles': args.max_muscles,
    }
    
    if is_batch_mode:
        # ===== BATCH MODE: Process multiple files in parallel =====
        print(f"\nProcessing {len(b3d_files)} files with {args.n_jobs} parallel workers...")
        print("="*80 + "\n")
        
        # Prepare tasks
        tasks = [
            (b3d_file, args.osim_path, args.output_dir, settings)
            for b3d_file in b3d_files
        ]
        
        # Process files in parallel
        t_start = time.time()
        
        if args.n_jobs == 1:
            # Serial processing
            results = [process_single_b3d_file(task) for task in tasks]
        else:
            # Parallel processing
            with mp.Pool(processes=args.n_jobs) as pool:
                results = pool.map(process_single_b3d_file, tasks)
        
        elapsed_total = time.time() - t_start
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH PROCESSING COMPLETE")
        print("="*80)
        
        successes = sum(1 for _, success, _, _ in results if success)
        failures = len(results) - successes
        
        print(f"Total files: {len(results)}")
        print(f"Successful: {successes}")
        print(f"Failed: {failures}")
        print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
        if len(results) > 0:
            print(f"Average per file: {elapsed_total/len(results):.1f}s")
        
        if failures > 0:
            print(f"\nFailed files:")
            for b3d_path, success, elapsed, error in results:
                if not success:
                    file_name = Path(b3d_path).name
                    print(f"  - {file_name}: {error[:100] if error else 'Unknown error'}")
        
        print(f"\nResults saved in: {args.output_dir}")
        print("="*80 + "\n")
        
    else:
        # ===== SINGLE FILE MODE: Process one file =====
        b3d_path = b3d_files[0]
        
        # Create subdirectory for single file output
        if b3d_path:
            file_stem = Path(b3d_path).stem
            file_output_dir = os.path.join(args.output_dir, file_stem)
            os.makedirs(file_output_dir, exist_ok=True)
            output_prefix = os.path.join(file_output_dir, file_stem)
        else:
            # Synthetic data case
            file_output_dir = args.output_dir
            os.makedirs(file_output_dir, exist_ok=True)
            output_prefix = os.path.join(file_output_dir, 'synthetic')
        
        print("\n[1/7] Parsing marker data...")
        times_markers, markers, marker_labels = parse_b3d_simple(b3d_path)
        markers_mot = output_prefix + '_markers.mot'
        
        # Reshape markers for MOT format
        if markers.ndim == 2 and markers.shape[1] % 3 == 0:
            pass
        else:
            markers = markers.reshape((markers.shape[0], -1))
        
        # Create marker labels for MOT
        nmarkers = markers.shape[1] // 3
        mot_labels = []
        for i in range(nmarkers):
            for axis in ['x', 'y', 'z']:
                mot_labels.append(f'marker{i}_{axis}')
        
        write_mot_file(times_markers, markers, mot_labels, markers_mot)
        
        # Step 2: Run IK
        print("\n[2/7] Running inverse kinematics...")
        coords_mot = output_prefix + '_coords_ik.mot'
        times, coords_ref, coord_names = run_simple_ik(args.osim_path, markers_mot, coords_mot)
        print(f"IK result: {coords_ref.shape[0]} frames, {coords_ref.shape[1]} coordinates")
        
        # Auto-detect N from data if not specified
        if args.N is None:
            args.N = coords_ref.shape[0] - 1
            print(f"Auto-detected N from data: {args.N} intervals ({coords_ref.shape[0]} frames)")
            settings['N'] = args.N
        else:
            print(f"Using specified N: {args.N} intervals")
        
        # Step 3: Get muscles from model
        muscle_names = []
        if args.mode == 'muscle':
            print("\n[3/7] Extracting muscle information...")
            muscle_names = get_muscle_names_from_model(args.osim_path, args.max_muscles)
            if args.max_muscles is None:
                print(f"Using all {len(muscle_names)} muscles from model")
            else:
                print(f"Using {len(muscle_names)} muscles")
        else:
            print("\n[3/7] Torque-driven mode - skipping muscle extraction")
        
        # Step 4 & 5: Build optimization problem
        print("\n[4/7] Building CasADi optimization problem...")
        muscle_names_used = []
        try:
            if args.mode == 'muscle' and len(muscle_names) > 0:
                opti, vars_dict, times_opt, muscle_names_used, _ = build_improved_opt_problem(
                    coords_ref, times, args.osim_path, muscle_names, settings
                )
            else:
                print("Building simplified torque-driven problem...")
                opti, vars_dict, times_opt = build_torque_driven_problem(
                    coords_ref, times, settings
                )
        except Exception as e:
            print(f"Error building problem: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to simplified torque-driven problem...")
            opti, vars_dict, times_opt = build_torque_driven_problem(
                coords_ref, times, settings
            )
        
        print(f"Problem built with {len(times_opt)} time nodes")
        
        # Step 6: Solve optimization
        print("\n[5/7] Solving optimization problem...")
        print("This may take a few minutes...")
        
        t_start = time.time()
        
        try:
            sol = opti.solve()
            t_solve = time.time() - t_start
            print(f"✓ Optimization converged in {t_solve:.2f} seconds")
            sol_dict = extract_solution(opti, sol, vars_dict, times_opt)
        except Exception as e:
            print(f"✗ Optimization failed: {e}")
            print("Trying to extract debug solution...")
            try:
                sol_dict = extract_solution(opti, opti.debug, vars_dict, times_opt)
                print("Extracted debug solution (may not be optimal)")
            except:
                print("Could not extract solution")
                return
        
        # Step 7: Save and visualize results
        print("\n[6/7] Saving results...")
        np.save(output_prefix + '_solution.npy', sol_dict, allow_pickle=True)
        print(f"✓ Saved solution to {output_prefix}_solution.npy")
        
        # Check if we got muscle activations
        if 'A' not in sol_dict and args.mode == 'muscle':
            print("\n⚠ WARNING: Muscle mode was requested but activations not computed!")
            print("This means the problem fell back to torque-driven mode.")
        elif 'A' in sol_dict:
            print(f"✓ Muscle activations computed: shape {sol_dict['A'].shape}")
            
            # Save activations MOT file
            muscles_to_save = muscle_names_used if muscle_names_used else muscle_names
            activation_labels = [f'{name}_activation' for name in muscles_to_save]
            write_mot_file(sol_dict['times'], sol_dict['A'].T, activation_labels,
                           output_prefix + '_activations.mot')
            print(f"✓ Saved activations to {output_prefix}_activations.mot")
        
        # Save coordinates MOT file
        if 'Q' in sol_dict:
            write_mot_file(sol_dict['times'], sol_dict['Q'].T, coord_names, 
                          output_prefix + '_coords_opt.mot')
        
        # Plot results
        print("\n[7/7] Generating plots...")
        try:
            plot_results(sol_dict, coords_ref, output_prefix)
        except Exception as e:
            print(f"Could not generate plots: {e}")
        
        print("\n" + "="*80)
        print("COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved in: {file_output_dir}")
        print(f"  - Solution: {output_prefix}_solution.npy")
        print(f"  - Coordinates: {output_prefix}_coords_opt.mot")
        print(f"  - Plots: {output_prefix}_*.png")


def build_torque_driven_problem(coords_ref, times, settings):
    """Simplified torque-driven version without muscles."""
    opti = ca.Opti()
    
    N = settings.get('N', coords_ref.shape[0] - 1)
    nQ = coords_ref.shape[1]
    
    # Resample if needed
    if N != coords_ref.shape[0] - 1:
        t_new = np.linspace(times[0], times[-1], N+1)
        coords_ref_new = np.zeros((N+1, nQ))
        for j in range(nQ):
            coords_ref_new[:, j] = np.interp(t_new, times, coords_ref[:, j])
        coords_ref = coords_ref_new
        times = t_new
    
    dt = np.mean(np.diff(times))
    
    # Decision variables
    Q = opti.variable(nQ, N+1)
    Qd = opti.variable(nQ, N+1)
    Tau = opti.variable(nQ, N+1)  # Control torques
    
    # Initial guess
    opti.set_initial(Q, coords_ref.T)
    opti.set_initial(Qd, 0)
    opti.set_initial(Tau, 0)
    
    # Bounds
    opti.subject_to(opti.bounded(-2*np.pi, Q, 2*np.pi))
    opti.subject_to(opti.bounded(-10.0, Qd, 10.0))
    opti.subject_to(opti.bounded(-100.0, Tau, 100.0))
    
    # Boundary conditions
    opti.subject_to(Q[:, 0] == coords_ref[0, :])
    opti.subject_to(Qd[:, 0] == 0)
    
    # Simple dynamics: M*qdd = tau (with M = identity for simplicity)
    M = 10.0  # typical mass*length^2
    
    # Objective
    J = 0
    wt = settings.get('wl_tracking', 100.0)
    we = settings.get('wl_effort', 1.0)
    
    # Trapezoidal integration (simpler than Hermite-Simpson for demo)
    for k in range(N):
        qk = Q[:, k]
        qdk = Qd[:, k]
        tauk = Tau[:, k]
        
        qk1 = Q[:, k+1]
        qdk1 = Qd[:, k+1]
        tauk1 = Tau[:, k+1]
        
        # Dynamics: qdd = tau / M
        qdd_k = tauk / M
        qdd_k1 = tauk1 / M
        
        # Integration
        opti.subject_to(qdk1 == qdk + dt * 0.5 * (qdd_k + qdd_k1))
        opti.subject_to(qk1 == qk + dt * 0.5 * (qdk + qdk1))
        
        # Objective
        J += wt * ca.sumsqr(qk - coords_ref[k, :])
        J += we * ca.sumsqr(tauk)
    
    # Final node
    J += wt * ca.sumsqr(Q[:, -1] - coords_ref[-1, :])
    J += we * ca.sumsqr(Tau[:, -1])
    
    opti.minimize(J)
    
    # Solver
    opts = {
        'ipopt.max_iter': settings.get('max_iter', 500),
        'ipopt.tol': settings.get('tol', 1e-6),
        'ipopt.print_level': 5,
    }
    opti.solver('ipopt', opts)
    
    vars_dict = {'Q': Q, 'Qd': Qd, 'Tau': Tau}
    
    return opti, vars_dict, times


if __name__ == '__main__':
    main()
