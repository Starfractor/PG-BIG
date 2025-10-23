import os
import json
from tqdm import tqdm

import torch
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_limo as option_limo
import utils.utils_model as utils_model
from dataset import dataset_MOT_MCS, dataset_MOT_segmented

from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nimblephysics as nimble 
from classifiers import get_classifier
from write_mot import write_mot33, write_mot35, write_mot33_simulation
from collections import OrderedDict
import deepspeed 
from profile_encoder import ProfileEncoder, ProfileDecoder 
import torch.nn.functional as F
from decoder_dataset import generate_test_profiles
latent_dim = 16
profile_dim = 512
biomech_dim = 2048
seq_len = 512
num_actions = 38
torch.set_grad_enabled(True)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.autograd.set_detect_anomaly(True)


################ Guidance extraction helpers ################
def _safe_array(x):
    try:
        return np.asarray(x)
    except Exception:
        return None

def extract_guidance_from_b3d(b3d_path, trial_idx=0, ref_frame=0, plate_units_in_mm=True):
    """Best-effort extraction of guidance signals from a .b3d file's trial.

    Returns a dict with keys:
      - pelvic_tilt: (T,) radians
      - angular_velocity: (T,) magnitude (rad/s) estimated from pelvis orientation
      - com_vel: (T,3)
      - com_acc: (T,3)
      - foot_contact: (T, n_plates) boolean or force magnitude
      - muscle_activation: (T, M) predicted by surrogate (if available)
    """
    out = {}
    try:
        subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
        tlen = subj.getTrialLength(trial_idx)
        timestep = subj.getTrialTimestep(trial_idx)
        frames = subj.readFrames(trial=trial_idx, startFrame=0, numFramesToRead=tlen, includeProcessingPasses=True, includeSensorData=True)

        # Pelvis orientation (try to read pose pos/rot from kinematics pass)
        pelvis_angles = []
        pelvis_av = []
        com_vel = []
        com_acc = []
        foot_forces = []

        for frame in frames:
            # processingPasses[0] is KINEMATICS usually
            pp = frame.processingPasses[0] if frame.processingPasses else None
            if pp is not None:
                # pelvis tilt: try pos or rotation fields
                try:
                    # pos is flat array of joint angles; find pelvis tilt index via skeleton? fallback: 0
                    if hasattr(pp, 'pos'):
                        pos = np.array(pp.pos)
                        pelvis_angles.append(pos[0] if pos.size>0 else 0.0)
                    else:
                        pelvis_angles.append(0.0)
                except Exception:
                    pelvis_angles.append(0.0)

                try:
                    gv = getattr(pp, 'comVel', None)
                    ca = getattr(pp, 'comAcc', None)
                    com_vel.append(np.array(gv) if gv is not None else np.zeros(3))
                    com_acc.append(np.array(ca) if ca is not None else np.zeros(3))
                except Exception:
                    com_vel.append(np.zeros(3))
                    com_acc.append(np.zeros(3))

                # foot contact: groundContactForce if present
                try:
                    grf = getattr(pp, 'groundContactForce', None)
                    if grf is None:
                        foot_forces.append(np.zeros(0))
                    else:
                        foot_forces.append(np.array(grf).reshape(-1,3))
                except Exception:
                    foot_forces.append(np.zeros(0))
            else:
                pelvis_angles.append(0.0)
                com_vel.append(np.zeros(3))
                com_acc.append(np.zeros(3))
                foot_forces.append(np.zeros(0))

        pelvis_angles = np.array(pelvis_angles)
        com_vel = np.vstack(com_vel) if len(com_vel)>0 else np.zeros((tlen,3))
        com_acc = np.vstack(com_acc) if len(com_acc)>0 else np.zeros((tlen,3))

        # angular velocity: approximate via finite differences of pelvis angle
        ang_vel = np.zeros_like(pelvis_angles)
        if timestep and len(pelvis_angles)>1:
            ang_vel[1:] = np.diff(pelvis_angles) / timestep

        # foot contact binary from force magnitude
        foot_contact = []
        for ff in foot_forces:
            if ff.size == 0:
                foot_contact.append(np.zeros(1))
            else:
                mag = np.linalg.norm(ff, axis=1) if ff.ndim==2 else np.array([np.linalg.norm(ff)])
                foot_contact.append(mag)
        # pad/conform foot_contact into array (num_plates x T)
        # determine max plates per frame
        maxp = max([fc.shape[0] if hasattr(fc,'shape') else 0 for fc in foot_forces]) if foot_forces else 0
        if maxp == 0:
            foot_contact_arr = np.zeros((tlen,0))
        else:
            arr = np.zeros((tlen,maxp))
            for i,fc in enumerate(foot_forces):
                if fc.size == 0:
                    continue
                mag = np.linalg.norm(fc, axis=1) if fc.ndim==2 else np.array([np.linalg.norm(fc)])
                L = min(len(mag), maxp)
                arr[i,:L] = mag[:L]
            foot_contact_arr = arr

        out['pelvic_tilt'] = pelvis_angles
        out['angular_velocity'] = ang_vel
        out['com_vel'] = com_vel
        out['com_acc'] = com_acc
        out['foot_contact'] = foot_contact_arr

        # Muscle activation via surrogate model (if available): use kinematic pose data
        try:
            if 'surrogate' in globals() and surrogate is not None:
                # build input kinematic tensor from frames using pp.pos
                kin = []
                for frame in frames:
                    pp = frame.processingPasses[0] if frame.processingPasses else None
                    if pp is not None and hasattr(pp, 'pos'):
                        kin.append(np.array(pp.pos))
                    else:
                        kin.append(np.zeros(IN_T*IN_D if 'IN_T' in globals() else 33))
                kin = np.vstack(kin)
                kin_t = torch.from_numpy(kin).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    acts = surrogate.predict(kin_t)
                out['muscle_activation'] = acts.squeeze(0).cpu().numpy()
        except Exception:
            out['muscle_activation'] = None

    except Exception as e:
        print(f"extract_guidance_from_b3d failed for {b3d_path}:{trial_idx}: {e}")
        return {}
    return out

def write_mot(path, data, framerate=60):
    header_string = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns=36\ninDegrees=yes\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\ntime	pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r	hip_adduction_r	hip_rotation_r	knee_angle_r	knee_angle_r_beta	ankle_angle_r	subtalar_angle_r	mtp_angle_r	hip_flexion_l	hip_adduction_l	hip_rotation_l	knee_angle_l	knee_angle_l_beta	ankle_angle_l	subtalar_angle_l	mtp_angle_l	lumbar_extension	lumbar_bending	lumbar_rotation	arm_flex_r	arm_add_r	arm_rot_r	elbow_flex_r	pro_sup_r	arm_flex_l	arm_add_l	arm_rot_l	elbow_flex_l	pro_sup_l\n"

    with open(path, 'w') as f:
        f.write(header_string)
        for i,d in enumerate(data):
            d = [str(i/60)] + [str(x) for x in d]
            
            # print(d)
            d = "      " +  "\t     ".join(d) + "\n"
            # print(d)
            f.write(d)

##### ---- Exp dirs ---- #####
args = option_limo.get_args_parser()
torch.manual_seed(args.seed)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark=False


args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)


args.out_dir = os.path.join(os.getcwd(), args.out_dir)

log_dir = os.path.join(args.out_dir,"logs")
os.makedirs(log_dir, exist_ok=True)
##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(log_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))








from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

action_to_desc = {  # replace
        "bend and pull full" : 0,
        "countermovement jump" : 1,
        "left countermovement jump" : 2,
        "left lunge and twist" : 3,
        "left lunge and twist full" : 4,
        "right countermovement jump" : 5,
        "right lunge and twist" : 6,
        "right lunge and twist full" : 7,
        "right single leg squat" : 8,
        "squat" : 9,
        "bend and pull" : 10,
        "left single leg squat" : 11,
        "push up" : 12
    }

'''
wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
'''

##### ---- Dataloader ---- #####
'''
args.nb_joints = 37 

val_loader = dataset_MOT_segmented.DATALoader(args.dataname,
                                        1,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        mode='limo')
'''


##### ---- Device ---- #####



##### ---- Network ---- #####
# NEW: import your exported notebook code and build encoder/decoder
from profile_encoder import ProfileEncoder, ProfileDecoder  # <-- change to actual class/function names from your notebook
from encdec_adapter import ProfileAdapter

# Build your models and load weights as you do in the notebook.
# Example (adjust to your API):
encoder = ProfileEncoder(
    latent_dim=latent_dim,
    profile_dim=profile_dim,
    bio_out_dim=64,
    biomech_dim=biomech_dim
).to(device)
encoder.load_state_dict(torch.load("profile_encoder.pth", map_location=device))


decoder = ProfileDecoder(profile_dim, latent_dim, seq_len, num_actions).to(device)
decoder.load_state_dict(torch.load("profile_decoder.pth", map_location=device))


adapter = ProfileAdapter(encoder, decoder, device=str(device))


from dataset.dataset_183_retarget import retargeted183_data_loader

# Load dataset
val_loader = retargeted183_data_loader(data_dir='/home/mnt/datasets/183_retargeted', num_workers=12, pre_load=True)


'''

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)
 
assert args.vq_name is not None, "Cannot run the optimization without a trained VQ-VAE"
logger.info('loading checkpoint from {}'.format(args.vq_name))
ckpt = torch.load(args.vq_name, map_location='cpu')
new_state_dict = {k.replace("module.", ""): v for k, v in ckpt['net'].items()}
net.load_state_dict(new_state_dict, strict=True)
net.eval()
net.to(device)
'''

############ Module to load subject info ################
MCS_PATH = "/data/panini/MCS_DATA"
args.subject = os.path.join(MCS_PATH, 'Data', args.subject) if not os.path.exists(args.subject) else args.subject


from osim_sequence import load_osim, groundConstraint, GetLowestPointLayer
'''
assert os.path.isdir(args.subject), "Location to subject info does not exist"
osim_path = os.path.join(args.subject,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted_contacts.osim')
assert os.path.exists(osim_path), f"Osim file:{osim_path} does not exist"
osim_geometry_dir = os.path.join("/data/panini/MCS_DATA",'OpenCap_LaiArnoldModified2017_Geometry')
assert os.path.exists(osim_geometry_dir), f"Osim geometry path:{osim_geometry_dir} does not exist"
osim = 
osim = load_osim(osim_path, osim_geometry_dir, ignore_geometry=False)
'''
rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
osim: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
subject_session = os.path.basename(args.subject.rstrip('/'))



################# Save location ########################
run = args.num_runs
save_folder = os.path.join("latents_subject","run_" + subject_session)
save_folder_mot = os.path.join(args.out_dir, "mot_visualization", "latents_subject_" + "run_" + subject_session)
save_folder_mot = os.path.join(os.getcwd(), save_folder_mot)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
if not os.path.exists(save_folder_mot):
    os.makedirs(save_folder_mot)

print("Save folder:",save_folder)
print("Save folder mot:",save_folder_mot)


################ Load Surrogate Model for muscle activity prediction ################
from surrogate import TransformerModel
surrogate = TransformerModel(input_dim=33, output_dim=80, num_layers=3, num_heads=3, dim_feedforward=128, dropout=0.1).to(device)
# Save path for the model
save_path = "transformer_surrogate_model_v2.pth"

assert os.path.exists(save_path), f"Model not found at {save_path}" 

surrogate.load_model(save_path)
surrogate.eval()


# Assert data is being loaded is compatible with nimble physics engine   
# for i,batch in tqdm(enumerate(val_loader)):
#     print("Testing ground constraint for batch:",i)
#     motions, m_lengths, names = batch
#     indices_to_keep = [i for i in range(motions[0].shape[2]) if i not in [10,18]]   
    
#     # for m_index, motion in enumerate(motions): 
#     motion = motions[0]
#     m= motion[:,:,indices_to_keep]
#     x = groundConstraint(osim, torch.tensor(m).cpu())
    
#     assert x.item() < 5, f"Person cannot fly, average lowest point on the skeleton above 5 meters.:{names[m_index]}"
        

import torch
from models import vqvae

# Load VQ-VAE Model
class Args:
    dataname = "mcs"
    nb_code = 512
    code_dim = 512
    output_emb_width = 512
    down_t = 2
    stride_t = 2
    width = 512
    depth = 3
    dilation_growth_rate = 3
    vq_act = "relu"
    vq_norm = None
    quantizer = "ema_reset"
    mu = 0.99
    nb_joints = 37
    batch_size = 32
    seq_len = 512
    lr = 0.01
    total_iter = 2000
args = Args()

# Instantiate the model
net = vqvae.HumanVQVAE(
    args,
    nb_code=args.nb_code,
    code_dim=args.code_dim,
    output_emb_width=args.output_emb_width,
    down_t=args.down_t,
    stride_t=args.stride_t,
    width=args.width,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation=args.vq_act,
    norm=args.vq_norm
)

# Set the device



ckpt = torch.load("/home/mnt/code/nicholas/output/VQVA11/300000.pth", map_location=device)

# Remove 'module.' prefix if present
state_dict = ckpt['net']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

net.load_state_dict(new_state_dict, strict=True)
net.eval()
net.to(device)
vqencoder = net.vqvae.encoder


def generate_train_embeddings():
    
    print("Generating Embeddings for proximity loss at: ", 'embeddings')
    os.makedirs('embeddings',exist_ok=True)

    data_dict = dict()

    for i,batch in tqdm(enumerate(val_loader)):
        ##################### OLD
        # motion, m_length, name = batch
        # # print(i,motion.shape, name)
        # motion = motion.cuda()
        # # print("motion shape:", motion.shape)
        # m = net.vqvae.preprocess(motion)
        # # print("m shape:", m.shape)
        # emb = net.vqvae.encoder(m)
        # # print("emb shape:", emb.shape)
        # # emb_proc = net.vqvae.postprocess(emb)
        # # print("emb proc shape:", emb_proc.shape)
        
        # emb = torch.squeeze(emb)
        # # emb = torch.transpose(emb,0,1)
        # emb = emb.cpu().detach().numpy()
        # # print(emb.shape)
        # # for j in range(emb.shape[0]):
        # #     data_dict["squat"].append(emb[i])
        # data_dict["squat"].append(emb)
        ###########################
        
    motions, m_lengths, action, names = batch
        print(names)
        motions.to(device)
        motions = net.vqvae.encoder(net.vqvae.preprocess(motions).to(device))
        # print(i,motion.shape, name)
        # motions = motions.cuda()
        for motion in motions:
            motion = motion.cuda()
            
            emb = adapter.encode(motion)           # -> (B, C, T)
            #print("emb shape:", emb.shape)
            emb = torch.squeeze(emb, dim=0)        # -> (C, T)  (for saving as a single sample)
            #print("look:", emb.shape)
            emb = emb.cpu().detach().numpy()
            # Attempt to also extract guidance constraints for this motion
            guidance = None
            try:
                dataset_obj = val_loader.dataset
                # Find the matching motion_info entry by name and subject
                subject_id = None
                try:
                    # val_loader yields names and action; dataset mapping contains subject_id
                    # Try to find mapping entry that matches both name and action
                    for entry in dataset_obj.motion_mapping:
                        if entry.get('name') == names[0] and entry.get('subject_id') == entry.get('subject_id'):
                            motion_info = entry
                            break
                    else:
                        # fallback to first mapping with same name
                        motion_info = next((e for e in dataset_obj.motion_mapping if e.get('name') == names[0]), None)
                except Exception:
                    motion_info = None

                if motion_info is not None:
                    b3d_file = motion_info.get('file_path')
                    trial_idx = motion_info.get('trial_idx')
                    guidance = None
                    try:
                        guidance = extract_guidance_from_b3d(b3d_file, trial_idx)
                    except Exception as e:
                        print(f"Failed to extract guidance for {b3d_file} trial {trial_idx}: {e}")
            except Exception:
                guidance = None

            data_dict[str(names[0])+"-"+action[0]] = [emb, guidance]

    # os.makedirs(os.path.join(args.out_dir, 'embeddings'), exist_ok = True)
    for k,v in data_dict.items():
        if len(v) == 0:
            continue
        array = np.array(v)
        np.save("embeddings/"+k+".npy",array)

generate_train_embeddings()

def load_train_embeddings(directory='embeddings'):
    # directory = os.path.join(args.out_dir, 'embeddings')
        
    embedding_dict = {}
    
    if not os.path.exists('embeddings') or len(os.listdir('embeddings')) == 0:
        generate_train_embeddings()
    
    for filename in os.listdir('embeddings'):
        if filename.endswith(".npy"):
            key = filename.split('.')[0]
            print(key)
            embedding = np.load('embeddings/'+filename)
            if len(embedding)==0:
                continue
            key.split('-')[0]
            embedding_dict[action_to_desc[key.split('-')[1]]] = torch.tensor(embedding)

    return embedding_dict

# Generate mot reconstruction for training data
# with torch.no_grad():
#     for i,batch in enumerate(val_loader):
#         motion, m_length, name = batch
#         out,_,_ = net(motion[:,:m_length[0]].cuda())
#         pred = out
#         write_mot('train_forward_pass/mot_output/'+str(i)+".npy", pred[0,:,:].detach().cpu().numpy())
#         out = out.squeeze(0).cpu().detach().numpy()
#         np.save('train_forward_pass/model_output/'+str(i)+".npy", out)
#         print(out.shape, np.array(motion).shape)

embedding_dict = load_train_embeddings()
print("Completing loading training embeddings:")
for k,v in embedding_dict.items():
    print(k,v.shape)

# def load_train_embeddings(directory='embeddings'):
#     # directory = os.path.join(args.out_dir, 'embeddings')
        
#     embedding_dict = {}
    
#     if not os.path.exists('embeddings') or len(os.listdir('embeddings')) == 0:
#         generate_train_embeddings()
    
#     for filename in os.listdir('embeddings'):
#         if filename.endswith(".npy"):
#             key = filename.split('.')[0]
#             embedding = np.load('embeddings/'+filename)
#             if len(embedding)==0:
#                 continue
            
#             embedding_dict[action_to_desc[key]] = embedding
    
#     return embedding_dict

def decode_latent(net, x_d):
    # x_d = x_d.permute(0, 2, 1).contiguous().float()
    x_quantized, _, _ = net.vqvae.quantizer(x_d)
    x_decoder = net.vqvae.decoder(x_quantized)
    x_out = x_decoder.permute(0, 2, 1)

    return x_out

def get_proximity_loss(z, embedding, reduce = True, chunk_size = 1000): #< 10 - 3
    
    batch_size = z.shape[0]
    num_embeddings = embedding.shape[0]

    min_distances = torch.full((batch_size,), float('inf'), device=z.device)
    min_indices = torch.zeros(batch_size, dtype=torch.long, device=z.device)

    for i in range(batch_size):
        # Initialize to keep track of the minimum distance and index for this batch
        min_dist = float('inf')
        min_idx = -1

        # Process embedding in chunks to avoid memory overload
        for start in range(0, num_embeddings, chunk_size):
            end = min(start + chunk_size, num_embeddings)
            embedding_chunk = embedding[start:end]

            # Compute distances for the chunk, keep dims (1, 2) as in original
            distances = torch.norm(z[i].unsqueeze(0) - embedding_chunk, dim=(1, 2))

            # Find the minimum distance and corresponding index in the chunk
            chunk_min_dist, chunk_min_idx = torch.min(distances, dim=0)

            # Update overall minimum distance and index if chunk's min is smaller
            if chunk_min_dist < min_dist:
                min_dist = chunk_min_dist
                min_idx = start + chunk_min_idx  # Adjust index for chunk offset
        
        # Store the final minimum distance and index for this batch
        min_distances[i] = min_dist
        min_indices[i] = min_idx

    # Sum the minimum distances
    if reduce:
        proximity_loss = min_distances.mean()
    else:
        proximity_loss = min_distances

    return proximity_loss, min_indices

def constained_optimization(x, low, high):

    # Compute the three expressions
    expr1 = x - high
    expr2 = low - x
    expr3 = torch.zeros_like(x)
    
    # Compute the element-wise maximum of the three expressions
    result = torch.max(torch.max(expr1, expr2), expr3)
    return result        
    
    
def get_optimized_z(category=9,initialization='mean', device='cuda'): 
    print("Optimizing for category:",category)
    print("Initialization:", embedding_dict)
    headers2indices = {osim.getDofByIndex(i).getName(): i for i in range(osim.getNumDofs())}

    # squat_muscles_indices = [val_loader.dataset.headers2indices[k] for k in val_loader.dataset.headers2indices if 'vaslat' in k or 'vasmed' in k] # Thigh muscles, (left and right) Vasus lateralis, medialis, intermedius
    squat_muscles_indices = [headers2indices[k] for k in headers2indices if 'vasmed' in k] # Thigh muscles, (left and right) Vasus lateralis, medialis, intermedius
    print("Squat muscles indices:",squat_muscles_indices)
    pelvis_tilt_index = headers2indices['pelvis_tilt']
    print("Pelvis tilt index:",pelvis_tilt_index)
    

    # Symmetry conditions
    symm_left_indices = ['hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
    symm_left_indices = [headers2indices[k] for k in symm_left_indices]
    symm_right_indices = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
    symm_right_indices = [headers2indices[k] for k in symm_right_indices]

    if initialization == 'random':
        # z = np.random.rand(args.batch_size, args.nb_code, args.seq_len).astype(np.float32)
        # z = torch.from_numpy(z).to(device)
        z_initial = torch.randn(args.batch_size, args.nb_code, args.seq_len, device=device,requires_grad=True)
        mult1 = np.random.uniform(50,120)
        mult2 = np.random.uniform(1,10)
        noise = torch.randn_like(z_initial) * mult2
        z = (z_initial * mult1 + noise).detach()
    elif initialization == 'mean':
        mean = torch.tensor(embedding_dict[category][0].unsqueeze(0).to(device),device=device).mean(dim=0)
        std = torch.tensor(embedding_dict[category][0].unsqueeze(0).to(device),device=device).std(dim=0)
        z = torch.randn(len(embedding_dict[category][0]), args.nb_code, device=device)
        # z = z * std + mean
        z = mean + torch.randn_like(z) * std #500
        z.requires_grad = True
    elif initialization == 'profile':
        profiles, subject_profiles = generate_test_profiles(encoder, val_loader.dataset, val_loader.dataset.subject_latents, [list(embedding_dict.keys())[category]], device)
        z = torch.tensor(profiles, device=device)
        z.requires_grad = True
    # z_np = z.detach().cpu().numpy()
    embedding = torch.tensor(embedding_dict[category][0].unsqueeze(0).to(device),device=device)
    proximity_embedding = torch.tensor(embedding_dict[category],device=device)
    loss_proximity = get_proximity_loss(z, proximity_embedding)



    # Define the second derivative kernel (Laplacian kernel)
    # Reshape and repeat the kernel for depthwise convolution
    com_acc_laplacian = torch.tensor([[1, -2, 1]], dtype=torch.float32, device=device)
    com_acc_laplacian = com_acc_laplacian.view(1, 1, -1)  # Shape: (1, 1, 3)
    com_acc_laplacian = com_acc_laplacian.repeat(3, 1, 1)  # Shape: (3, 1, 3)
    com_acc_laplacian = com_acc_laplacian.to(device)

    


    z.requires_grad = True
        
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam([z], lr=args.lr)

    # data_mean = torch.from_numpy(val_loader.dataset.mean).to(device)
    # data_std = torch.from_numpy(val_loader.dataset.std).to(device)
    proximity = []
    constrain = []
    
    losses_dict = {}

    for epoch in tqdm(range(args.total_iter)):
        #old_z = torch.from_numpy(z.detach().cpu().numpy()).to(device)
        old_z = z.clone().detach()
        optimizer.zero_grad()
        print(z.shape)
        pred_motion = decode_latent(net, z)
        pred_motion.requires_grad_(True)

        # De-Normalize
        # pred_motion = pred_motion * data_std.view(1,1,-1) + data_mean.view(1,1,-1)
        loss_proximity,min_indices = get_proximity_loss(z, proximity_embedding)
        # print("Min indices:",min_indices)

        # loss_temp = torch.tensor([0.0],device=device)
        loss_temp = torch.mean((pred_motion[:,1:,:]-pred_motion[:,:-1,:])**2)
 
        

        # Symmetry loss
        # loss_symm = torch.tensor([0.0],device=device)
        loss_symm = torch.tensor(0.0, device=device) #torch.mean((pred_motion[:,:,symm_left_indices] - pred_motion[:,:,symm_right_indices])**2)

        foot_loss = torch.tensor([0.0],device=device)
        foot_sliding_loss = torch.tensor([0.0],device=device)
        motion = pred_motion[:,:]
        
        for i in range(pred_motion.shape[0]):
            # m = motion[i]
            # m_tensor = torch.tensor(m, dtype=torch.float32, device=device, requires_grad=True)
            m_tensor = pred_motion[i,:]
            # for j in range(1,pred_motion.shape[1],3):
            for rand_j in range(1,int(epoch*5/(args.total_iter+1)) if epoch < args.total_iter - 1000 else 0):
                j = random.randint(1,pred_motion.shape[1]-2)
                nimble_input_motion = m_tensor[j]
                nimble_input_motion[6:] = torch.deg2rad(nimble_input_motion[6:])
                nimble_input_motion[:3] = 0 # Set pelvis to 0
                nimble_input_motion = nimble_input_motion.cpu()
                x = GetLowestPointLayer.apply(osim.skeleton, nimble_input_motion).to(device)
                foot_loss += x**2
                
                nimble_input_motion = m_tensor[j+1]
                nimble_input_motion[6:] = torch.deg2rad(nimble_input_motion[6:])
                nimble_input_motion[:3] = 0 # Set pelvis to 0
                nimble_input_motion = nimble_input_motion.cpu()
                x_t1 = GetLowestPointLayer.apply(osim.skeleton, nimble_input_motion).to(device)
                foot_sliding_loss += (x_t1-x)**2



        # Reduce jitter in the translation
        # loss_temp_trans = torch.tensor([0.0],device=device)
        loss_temp_trans = torch.mean((pred_motion[:,1:,3:6]-pred_motion[:,:-1,3:6])**2)

        com_acc = F.conv1d(
            input=F.pad(pred_motion[:,:,3:6].permute(0,2,1), (1, 1), mode='replicate'),  # Pad to maintain sequence length
            weight=com_acc_laplacian,
            groups=3).permute(0,2,1)  # Shape: (batch_size, seq_len, 3)
        
        # loss_temp_com = torch.tensor([0.0],device=device)
        loss_temp_com = F.smooth_l1_loss(com_acc, torch.zeros_like(com_acc), beta=0.01)


        # Lumbar extension constraint
        loss_tilt = torch.mean(pred_motion[:,:,pelvis_tilt_index]**2)
        # loss_tilt = torch.tensor([0.0],device=device)
        


        # Surrogate model loss
        #pred_muscle_activations = surrogate(pred_motion)        
        #surrogate_muscle_activation = torch.max(pred_muscle_activations[:,:,squat_muscles_indices],dim=1)[0]
        
        # constrain_loss = torch.tensor([0.0],device=device)
        #constrain_loss = constained_optimization(surrogate_muscle_activation,low=args.low,high=args.high)
        #constrain_loss = torch.sum(constrain_loss)

        # increase = True
        # if increase:
        #     surrogate_muscle_activation *= -1

        #surrogate_muscle_activation = torch.mean(surrogate_muscle_activation,dim=0)
        
        hyper_param_dict = {"proximity":0.01, \
            "tilt":0.001, "symmetry":1,\
            "foot":0.1, "foot_sliding":0.1,\
            "temporal":0.5, "temporal_trans":50, "com_acc":100,\
            "constrain":1}
        '''
        loss_dict = OrderedDict([["proximity", loss_proximity], \
            ["tilt", loss_tilt], ["symmetry", loss_symm], \
            ["foot", foot_loss], ["foot_sliding", foot_sliding_loss], \
            ["temporal", loss_temp], ["temporal_trans", loss_temp_trans], ["com_acc", loss_temp_com],\
            ["constrain", constrain_loss]])'''
        
        loss_dict = OrderedDict([["proximity", loss_proximity], \
            ["tilt", loss_tilt], ["symmetry", loss_symm], \
            ["foot", foot_loss], ["foot_sliding", foot_sliding_loss], \
            ["temporal", loss_temp], ["temporal_trans", loss_temp_trans], ["com_acc", loss_temp_com],\
            ])
        
        
        # loss_dict = OrderedDict([["proximity", 0.001*loss_proximity], \
        #     ["tilt", 0.001*loss_tilt], ["symmetry", loss_symm], \
        #     ["foot", foot_loss*0.1], ["foot_sliding", 0.1*foot_sliding_loss], \
        #     ["temporal", 0.5*loss_temp], ["temporal_trans", 50*loss_temp_trans], ["com_acc", 100*loss_temp_com],\
        #     ["constrain", constrain_loss]])                


        # pred_muscle_activations_thigh = pred_muscle_activations[:,:,:]

        # loss = loss_proximity * 0.0001
        # foot_loss = foot_loss.to(device)
        # foot_loss = torch.tensor(0,device=device)
        if epoch < 500 or epoch > args.total_iter - 1000: # Early start for proximity loss 
            # loss = loss_proximity * 0.001
            loss = hyper_param_dict["proximity"]*loss_dict["proximity"]
        elif epoch < 1000:
            loss = ((epoch-500)/500)*sum([hyper_param_dict[k]*loss_dict[k] for k in loss_dict if k != "proximity"])
            loss += loss_proximity * hyper_param_dict["proximity"] 
        else:
            # loss = loss_proximity * 0.001 \
            #     + loss_tilt * 0.001 + loss_symm \
            #     + foot_loss * 0.1 + 0.1*foot_sliding_loss \
            #     + 0.5*loss_temp + 50*loss_temp_trans \
            #     + constrain_loss 
            loss = sum([hyper_param_dict[k]*loss_dict[k] for k in loss_dict ])
        #loss = foot_loss*0.01
        loss.backward()
        # print("Z grad",z.grad)
        optimizer.step()
        
        loss_dict["loss"] = loss
        
        for k in loss_dict:
            loss_dict[k] = loss_dict[k].item()
            
        loss_dict['epoch'] = epoch
        loss_dict.move_to_end('epoch', last=False)
        loss_dict["Difference"] = torch.norm(z-old_z).item()
        
        hyper_param_dict["epoch"] = 1 
        hyper_param_dict["Difference"] = 1
        hyper_param_dict["loss"] = 1
        '''
        if epoch % 10 == 0:

            print(", ".join([str(k)+":"+f"{hyper_param_dict[k]*v if k in hyper_param_dict else v :6f}" for k,v in loss_dict.items()]))
            
            
            
            # print("Epoch:", epoch, "Loss:", loss.item(), \
            #     "Tilt Loss:", 0.001*loss_tilt.item(), "Symmetry Loss:", loss_symm.item(),
            #     "Penetration:", foot_loss.item()*0.1, "Sliding:", 0.1*foot_sliding_loss.item(), \
            #     "Temporal Loss:", 0.5*loss_temp.item(), "Proximity Loss:", 0.001*loss_proximity.item(), \
            #     "Trans Temporal:", 50*loss_temp_trans.item(), \
            #     f"Constrains:{constrain_loss.item()}")#"Difference:", torch.norm(z-old_z))
            
            per_muscle_avg_activation = [ (val_loader.dataset.indices2headers[squat_muscles_indices[i]],surrogate_muscle_activation[i].item())  for i in range(len(surrogate_muscle_activation))]
            print(f"Muscle activation:{per_muscle_avg_activation}")
            '''
        
        
        for k in loss_dict:
            if k not in losses_dict:
                losses_dict[k] = []
            losses_dict[k].append(loss_dict[k])
        
        
        if epoch % 1000 == 0 or epoch == args.total_iter-1:
            
            os.makedirs(os.path.join(save_folder_mot,f"save_LIMO/{subject_session}"),exist_ok=True)
            np.save(os.path.join(save_folder_mot,f"save_LIMO/{subject_session}/z_"+str(epoch)+".npy"),z.detach().cpu().numpy())
            np.save(os.path.join(save_folder_mot,f"save_LIMO/{subject_session}/pred_motion_"+str(epoch)+".npy"),pred_motion.detach().cpu().numpy())
        
            df = pd.DataFrame(losses_dict)
            df.to_csv(os.path.join(log_dir, f"losses_{subject_session}.csv"), index=False)

    ## SORT THE LATENTS BY THE LABELS
    
    with torch.no_grad():
        # z_quantized, _, _ = net.vqvae.quantizer(z)
        pred_motion = decode_latent(adapter, z)

        '''
        pred_muscle_activations = surrogate(pred_motion)        
        surrogate_muscle_activation = torch.amax(pred_muscle_activations[:,:,squat_muscles_indices],dim=(1,2))
        sort_indices = torch.argsort(surrogate_muscle_activation)
        '''
        # loss, min_indices = get_proximity_loss(z, proximity_embedding, reduce = False)
        # print(min_indices)
        # loss = loss.view(args.batch_size,-1) # Reshape to match sample x classifier window 
        # loss = loss.sum(1) # Sum across classifier windows      
        # sort_indices = torch.argsort(loss)
        # z = z[sort_indices]
        # loss = loss[sort_indices]
        
        z = z[sort_indices]
        loss = surrogate_muscle_activation[sort_indices]
        
        min_idx = min_indices[sort_indices]
        print("Sorted min indices:",min_idx, loss)

    del optimizer
    del loss_fn
    del proximity_embedding
    
    return z,loss

i = 9

if i == 9:
    # Added to run multiple iterations of LIMO for evaluation purposes
    # for run in range(args.num_runs):
    


    torch.cuda.empty_cache() # Clear cache to avoid extra memory usage

    category_name = "squat"
    # save_folder = os.path.join("latents",'category_'+category_name)    
    z,score = get_optimized_z(category=i)
    print(z.shape)
    decoded_z = decode_latent(adapter, z)

    bs = decoded_z.shape[0]
    bs = min(bs,args.min_samples)
    for j in range(bs):
        entry = decoded_z[j]
        file_path = os.path.join(save_folder,f'entry_{j}_{args.exp_name}.npy')
        print(f"Saving results in file:{file_path}")
        np.save(file_path, entry.cpu().detach().numpy())
        
        pred_motion_saved = np.load(file_path)
        mot_file_path = os.path.join(save_folder_mot,f'entry_{j}_{args.exp_name}.mot')
        write_mot33_simulation(mot_file_path, pred_motion_saved)
        print(f"Saving mot in file:{mot_file_path}")
    # np.save(os.path.join(args.out_dir,f'scores_{category_name}.npy'), score.cpu().detach().numpy())

    del z 
    del score
    del decoded_z





print(f"out_dir:{args.out_dir}")

import os
evaluater_path = os.path.join(os.getcwd(), "..", "UCSD-OpenCap-Fitness-Dataset")
os.chdir(evaluater_path)



os.system("/home/ubuntu/shareconda/etc/profile.d/conda.sh && conda activate T2M-GPT")

evaluate_log_path = os.path.join(log_dir, f'evaluate_{subject_session}.log')

print(f"Running command: conda run -n T2M-GPT python src/evaluate_retrieved_mot_files.py -m {os.path.join(args.out_dir,f'mot_visualization/latents_subject_run_{subject_session}')} -d /home/ubuntu/data/MCS_DATA/ --force > {evaluate_log_path}")
os.system(f"conda run -n T2M-GPT python src/evaluate_retrieved_mot_files.py -m {os.path.join(args.out_dir,f'mot_visualization/latents_subject_run_{subject_session}')} -d /home/ubuntu/data/MCS_DATA/ --force > {evaluate_log_path}")
os.system(f"cat {evaluate_log_path}")

foot_metrics_log_path = os.path.join(log_dir, f'foot_metrics_{subject_session}.log')

os.system(f"conda run -n T2M-GPT python src/evaluation/foot_sliding_checker.py --sample_dir  ../MCS_DATA/latents_subject_run_{subject_session}.txt > {foot_metrics_log_path}")
os.system(f"cat {foot_metrics_log_path}")


os.environ['DISPLAY'] = ':99.0'

mocap_motion_paths = f"/home/ubuntu/data/opencap-processing/Data/{subject_session}/MarkerData/"    
mocap_motion_paths = [os.path.join(mocap_motion_paths,file) for file in os.listdir(mocap_motion_paths) if file.endswith(".trc") and "SQT" in file.upper()]


for i in range(0, bs, 9): # Every 4th entry out of 20 / 5 samples sorted by muscle activation
    mot_file_path = os.path.join(save_folder_mot,f'entry_{i}_{args.exp_name}.mot')
    
    mocap_motion_path = mocap_motion_paths[i%len(mocap_motion_paths)]

    print(f"Running Command: conda run -n T2M-GPT python src/opencap_reconstruction_render.py {mocap_motion_path} {mot_file_path} {os.path.join(args.out_dir,'latest_rendered') }")
    os.system(f"conda run -n T2M-GPT python src/opencap_reconstruction_render.py {mocap_motion_path} {mot_file_path} {os.path.join(args.out_dir,'latest_rendered') }")

    os.system(f"rm  {os.path.join(args.out_dir,'latest_rendered/*/images/*')}") # Clear the images folder
