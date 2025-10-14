import argparse
import os
import numpy as np
import torch
import warnings
import torch.nn.functional as F

from collections import defaultdict
from dataset import dataset_183_retarget
from models import vqvae, profile_modules
import options.option_subject_prior as option_subject_prior


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ('net', 'state_dict', 'model'):
            if k in ckpt:
                sd = ckpt[k]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        if any(k.startswith('module.') for k in sd.keys()):
            new_sd = {}
            for k, v in sd.items():
                new_k = k[len('module.'):] if k.startswith('module.') else k
                new_sd[new_k] = v
            sd = new_sd
    return sd


def get_biomech_vector(subject, dataset, biomech_dim):
    bm = dataset.get_biomech(subject)
    if isinstance(bm, dict):
        vec = bm.get('vector', None)
        if vec is None:
            vec = dataset._biomech_to_vector(
                bm,
                target_samples=dataset.default_biomech_target_samples,
                biomech_dim=dataset.default_biomech_dim,
            )
    else:
        bm2 = dataset.get_biomech(subject)
        vec = bm2.get('vector', np.zeros(biomech_dim, dtype=np.float32))
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[0] != biomech_dim:
        if vec.shape[0] < biomech_dim:
            vec = np.pad(vec, (0, biomech_dim - vec.shape[0]), mode='constant')
        else:
            vec = vec[:biomech_dim]
    return vec


def build_subject_samples(loader, model, device, max_samples=10000):
    # Iterate the data loader and build mapping of subject -> list of (latent, action)
    subject_samples = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                motions, _, names, subject_names = batch
            except ValueError:
                continue
            motions = motions.to(device, dtype=torch.float32)
            latents = model.vqvae.encoder(model.vqvae.preprocess(motions))
            for latent, subj, act in zip(latents, subject_names, names):
                subject_samples[subj].append((latent.cpu(), act))
            if sum(len(v) for v in subject_samples.values()) >= max_samples:
                break
    return subject_samples


def main():
    parser = argparse.ArgumentParser(description='Inference: generate motion from profile decoder')
    parser.add_argument('--vqvae-checkpoint', type=str, default="./outputs/VQVAE/300000.pth")
    parser.add_argument('--encoder-checkpoint', type=str, default="./outputs/profile_encoder/profile_encoder.pth")
    parser.add_argument('--decoder-checkpoint', type=str, default="./outputs/subject_prior/profile_decoder.pth")
    parser.add_argument('--dataset-path', type=str, default='./dataset/183_retargeted')
    parser.add_argument('--subject', type=str, default=None, help='Subject name to use for profile (default: first available)')
    parser.add_argument('--subject-idx', type=int, default=None, help='Subject index (into dataset subjects) to use for profile (alternative to --subject)')
    parser.add_argument('--action', type=str, default=None, help='Action name to condition on (default: use sample action)')
    parser.add_argument('--action-idx', type=int, default=None, help='Action index to condition on (alternative to --action)')
    parser.add_argument('--condition-motion', type=str, default=None, help='Optional: path to a .npy motion file to encode and use as conditioning latent')
    parser.add_argument('--overlap-frames', type=int, default=None, help='Number of frames to overlap between windows when stitching a long conditioned motion')
    parser.add_argument('--hop-frames', type=int, default=None, help='Hop (stride) between windows when stitching; overrides overlap if set')
    parser.add_argument('--profile-file', type=str, default=None, help='Optional: path to a .npy file containing a precomputed profile vector to use instead of encoding')
    parser.add_argument('--num-actions', type=int, default=20, help='Number of actions (must match decoder)')
    parser.add_argument('--out-file', type=str, default='generated_motion.npy')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load VQ-VAE
    # Load VQ-VAE checkpoint; prefer weights_only=True when available to avoid
    # untrusted-pickle FutureWarning and reduce the pickle surface.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions may not support weights_only kwarg
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device)
    # Use either saved args from the checkpoint (preferred) or fall back to a
    # minimal dummy args.Namespace so the constructor receives expected attrs.
    if isinstance(vqvae_ckpt, dict) and 'args' in vqvae_ckpt:
        model_args = argparse.Namespace(**vqvae_ckpt['args'])
    else:
        model_args = argparse.Namespace(
        dataname='183_athletes',
        batch_size=128,
        window_size=512,
        local_rank=0,
        total_iter=100000,
        warm_up_iter=1000,
        lr=2e-4,
        lr_scheduler=[50000, 400000],
        gamma=0.05,
        temporal=2.0,
        weight_decay=0.0,
        commit=0.02,
        loss_vel=0.1,
        recons_loss='l2',
        code_dim=512,
        nb_code=512,
        mu=0.99,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        output_emb_width=512,
        vq_act='relu',
        vq_norm=None,
        quantizer='ema_reset',
        beta=1.0,
        resume_pth=None,
        resume_gpt=None,
        out_dir='output_vqfinal/',
        results_dir='visual_results/',
        visual_name='baseline',
        exp_name='exp_debug',
        print_iter=200,
        eval_iter=1000,
        seed=123,
        vis_gt=False,
        nb_vis=20,
        nb_joints=37,
    )
    model = vqvae.HumanVQVAE(
        model_args,
        nb_code=512, code_dim=512, output_emb_width=512,
        down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
        activation='relu', norm=None
    )
    sd = _extract_state_dict(vqvae_ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    # Debug: print VQ-VAE training args that determine temporal sizes
    try:
        ws = getattr(model_args, 'window_size', None)
        st = getattr(model_args, 'stride_t', None)
        dp = getattr(model_args, 'depth', None)
        print(f"VQ-VAE config: window_size={ws}, stride_t={st}, depth={dp}")
    except Exception:
        pass

    # Load profile encoder. If the encoder checkpoint contains saved args,
    # prefer those values; otherwise fall back to reasonable defaults.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    enc_args = enc_ckpt['args'] if isinstance(enc_ckpt, dict) and 'args' in enc_ckpt else {}
    latent_dim = enc_args.get('latent_dim', 16)
    profile_dim = enc_args.get('profile_dim', 512)
    bio_out_dim = enc_args.get('bio_out_dim', 64)
    biomech_dim_encoder = enc_args.get('biomech_dim', 2048)

    encoder = profile_modules.ProfileEncoder(
        latent_dim=latent_dim, profile_dim=profile_dim, bio_out_dim=bio_out_dim, biomech_dim=biomech_dim_encoder
    ).to(device)
    enc_sd = _extract_state_dict(enc_ckpt)
    encoder.load_state_dict(enc_sd, strict=False)
    encoder.eval()

    # Build dataset and samples
    loader = dataset_183_retarget.retargeted183_data_loader(data_dir=args.dataset_path, num_workers=4, pre_load=False)
    dataset = loader.dataset
    subject_samples = build_subject_samples(loader, model, device, max_samples=2000)
    if len(subject_samples) == 0:
        raise RuntimeError('No subject samples found in dataset')

    # Choose subject
    chosen_subject = args.subject
    if chosen_subject is None:
        chosen_subject = next(iter(subject_samples.keys()))
        print(f'No subject provided, using first subject: {chosen_subject}')
    if chosen_subject not in subject_samples:
        raise ValueError(f'Subject {chosen_subject} not found in dataset')

    # Pick first sample for that subject unless a conditioned motion is provided
    sample_latent, sample_action = subject_samples[chosen_subject][0]
    print(f'Using sample action: {sample_action} for subject: {chosen_subject}')

    # If user provided a conditioned motion, load and encode it to obtain latents
    if args.condition_motion is not None:
        cm_path = args.condition_motion
        if not os.path.isfile(cm_path):
            raise FileNotFoundError(f'Condition motion file not found: {cm_path}')
        cm = np.load(cm_path, allow_pickle=False)
        # Expect shape (T, joints) or (B, T, joints)
        if cm.ndim == 2:
            cm_t = torch.tensor(cm, device=device, dtype=torch.float32).unsqueeze(0)
        elif cm.ndim == 3:
            cm_t = torch.tensor(cm, device=device, dtype=torch.float32)
        else:
            raise ValueError('Condition motion .npy must be 2D (T, D) or 3D (B, T, D)')
        # If the conditioned motion is longer than a single VQ-VAE window,
        # split into overlapping windows, encode+quantize+decode each, and stitch
        # the decoded windows back together using overlap-add with a linear
        # crossfade envelope.
        cm_len = cm.shape[0]
        window_size = getattr(model_args, 'window_size', None) or 512
        # Determine hop/overlap
        if args.hop_frames is not None:
            hop = int(args.hop_frames)
            overlap = max(0, window_size - hop)
        else:
            overlap = int(args.overlap_frames) if args.overlap_frames is not None else window_size // 2
            hop = window_size - overlap

        def split_windows(array, win, hop):
            starts = list(range(0, max(1, array.shape[0] - win + 1), hop))
            # ensure last window covers the end
            if len(starts) == 0 or starts[-1] + win < array.shape[0]:
                starts.append(max(0, array.shape[0] - win))
            windows = [array[s:s+win] if s+win <= array.shape[0] else np.pad(array[s:], ((0, s+win-array.shape[0]), (0,0)), mode='constant') for s in starts]
            return windows, starts

        def stitch_windows(windows, starts, total_len):
            # windows: list of (L, D) arrays; starts: list of start indices
            D = windows[0].shape[1]
            out = np.zeros((total_len, D), dtype=np.float32)
            weight = np.zeros((total_len,), dtype=np.float32)
            L = windows[0].shape[0]
            fade = max(0, window_size - hop)
            for w, s in zip(windows, starts):
                env = np.ones((w.shape[0],), dtype=np.float32)
                if fade > 0:
                    f = fade
                    # linear fade in/out over 'f' samples
                    if w.shape[0] >= f:
                        lin = np.linspace(0.0, 1.0, f, endpoint=False, dtype=np.float32)
                        env[:f] = env[:f] * lin
                        env[-f:] = env[-f:] * lin[::-1]
                for i in range(w.shape[0]):
                    idx = s + i
                    if idx >= total_len:
                        break
                    out[idx] += w[i] * env[i]
                    weight[idx] += env[i]
            # avoid divide by zero
            nz = weight > 0
            out[nz] = out[nz] / weight[nz][:, None]
            return out

        # If the conditioned motion is shorter than window_size, behave as before
        if cm_len <= window_size:
            with torch.no_grad():
                try:
                    pre = model.vqvae.preprocess(cm_t)
                except Exception:
                    pre = cm_t
                latents = model.vqvae.encoder(pre)
                sample_latent = latents[0].cpu()
                print(f'Encoded conditioned motion from {cm_path} to latent (shape {sample_latent.shape})')
        else:
            # Split into windows and process each window through encoder->quantizer->decoder
            windows, starts = split_windows(cm, window_size, hop)
            decoded_windows = []
            with torch.no_grad():
                first_window = True
                for i, w in enumerate(windows):
                    wt = torch.tensor(w, device=device, dtype=torch.float32).unsqueeze(0)
                    try:
                        pre = model.vqvae.preprocess(wt)
                    except Exception:
                        pre = wt
                    lat = model.vqvae.encoder(pre)  # (B, code_dim, seq_len)
                    if first_window:
                        try:
                            print(f"Window {i}: encoder output shape: {tuple(lat.shape)}")
                        except Exception:
                            pass
                    # convert to (B, seq_len, code_dim)
                    lat_t = lat.transpose(1, 2)
                    B, seq_l, code_dim = lat_t.shape
                    flat = lat_t.reshape(B * seq_l, code_dim)
                    # quantize
                    try:
                        codes = model.vqvae.quantizer.quantize(flat)
                    except Exception:
                        # fallback if quantizer expects different input
                        codes = model.vqvae.quantizer.quantize(flat)
                    codes = codes.view(B, seq_l)
                    if first_window:
                        try:
                            print(f"Window {i}: codes shape: {tuple(codes.shape)} (B, seq_len)")
                        except Exception:
                            pass
                    # try decoding from indices if helper exists
                    dec = None
                    try:
                        if hasattr(model.vqvae, 'decode_from_indices'):
                            dec = model.vqvae.decode_from_indices(codes)
                        else:
                            # build embeddings and call decoder
                            codebook = model.vqvae.quantizer.codebook.to(device)
                            emb = codebook[codes.squeeze(0)]  # (seq_l, De)
                            emb_t = emb.unsqueeze(0).transpose(1, 2)  # (1, De, seq_l)
                            if hasattr(model.vqvae, 'decoder'):
                                try:
                                    dec = model.vqvae.decoder(emb_t)
                                except Exception:
                                    dec = model.vqvae.decoder(emb_t.transpose(1, 2))
                    except Exception as e:
                        print(f'Window {i} decode failed: {e}')
                        dec = None
                    if dec is None:
                        raise RuntimeError(f'Failed to decode window {i}')
                    # Normalize decoder output to (1, D, T)
                    if dec.ndim == 3 and dec.shape[1] == windows[0].shape[0]:
                        # dec is (1, T, D) -> transpose to (1, D, T)
                        dec_t = dec.transpose(1, 2)
                    else:
                        # assume dec is (1, D, T)
                        dec_t = dec

                    # dec_t: (1, D, T). Resize in time dimension to match window_size if necessary
                    decoded_T = dec_t.shape[2]
                    if decoded_T != window_size:
                        try:
                            dec_t_resized = F.interpolate(dec_t, size=window_size, mode='linear', align_corners=False)
                            print(f"Window {i}: resized decoded frames {decoded_T} -> {window_size}")
                        except Exception:
                            # fallback: keep original
                            dec_t_resized = dec_t
                    else:
                        dec_t_resized = dec_t

                    # Convert to (L, D)
                    dec_np = dec_t_resized.squeeze(0).cpu().numpy().transpose(1, 0)
                    if first_window:
                        try:
                            print(f"Window {i}: decoded window shape (frames, dims): {dec_np.shape}")
                        except Exception:
                            pass
                        first_window = False
                    decoded_windows.append(dec_np)
            # Stitch windows back to full length
            stitched = stitch_windows(decoded_windows, starts, cm_len)
            # Save stitched result and exit
            np.save(args.out_file, stitched)
            print(f'Stitched decoded motion saved to {args.out_file} (shape {stitched.shape})')
            return

    biomech_dim = getattr(dataset, 'default_biomech_dim', 2048)
    bio_vec = get_biomech_vector(chosen_subject, dataset, biomech_dim)

    # Compute or load profile vector for chosen subject sample
    if args.profile_file is not None:
        pf = args.profile_file
        if not os.path.isfile(pf):
            raise FileNotFoundError(f'Profile file not found: {pf}')
        profile_vec = np.load(pf)
        profile_vec = torch.tensor(profile_vec, device=device, dtype=torch.float32)
        print(f'Loaded profile vector from {pf} (shape {profile_vec.shape})')
    else:
        with torch.no_grad():
            latent_t = sample_latent.to(device).unsqueeze(0)
            bio_t = torch.tensor(bio_vec, device=device, dtype=torch.float32).unsqueeze(0)
            try:
                profile_vec = encoder(latent_t, bio_t)
            except TypeError:
                profile_vec = encoder(latent_t)
            profile_vec = profile_vec.squeeze(0)

    # Prepare action selection info (we'll build an action vector after
    # inferring the decoder's expected action embedding size)
    action_name = args.action if args.action is not None else sample_action
    # Build action list. Prefer action_list stored in decoder checkpoint if present
    # (ensures indices match training). Otherwise fall back to dataset-derived list.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dec_ckpt = torch.load(args.decoder_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dec_ckpt = torch.load(args.decoder_checkpoint, map_location=device)
    dec_sd = _extract_state_dict(dec_ckpt)

    ckpt_action_list = None
    if isinstance(dec_ckpt, dict) and 'action_list' in dec_ckpt:
        ckpt_action_list = dec_ckpt['action_list']

    if ckpt_action_list is not None:
        all_actions = list(ckpt_action_list)
    else:
        all_actions = sorted({act for samples in subject_samples.values() for (_, act) in samples})
    if action_name not in all_actions:
        print(f'Action {action_name} not found in dataset actions; using first action in list')
        action_name = all_actions[0]
    action_to_idx = {a: i for i, a in enumerate(all_actions)}
    # If action not found, fallback to index 0
    action_idx = action_to_idx.get(action_name, 0)

    # Instantiate mapping network (decoder)
    # Determine seq_len and code_embed_dim from sample latent and vqvae codebook
    sample_shape = sample_latent.shape
    # sample_latent shape is (code_dim, seq_len) in this codebase
    seq_len = sample_shape[1]
    code_embed_dim = model.vqvae.quantizer.codebook.shape[1]

    # Try to extract action embedding info from decoder checkpoint. If the
    # checkpoint saved an embedding state, use its shape. Otherwise infer from
    # the mapping network weights; finally fall back to args.num_actions.
    action_emb_dim = None
    action_embedding = None
    num_actions = None
    if isinstance(dec_ckpt, dict) and 'action_embedding' in dec_ckpt:
        emb_sd = dec_ckpt['action_embedding']
        # emb_sd is a state dict like {'weight': Tensor}
        weight = emb_sd.get('weight') if isinstance(emb_sd, dict) else None
        if weight is not None:
            num_actions, action_emb_dim = int(weight.shape[0]), int(weight.shape[1])
            # create embedding and load
            action_embedding = torch.nn.Embedding(num_actions, action_emb_dim).to(device)
            action_embedding.load_state_dict(emb_sd)
    if action_emb_dim is None:
        # fallback: inspect mapping weights for action_proj.weight
        for k, v in dec_sd.items():
            if k.endswith('action_proj.weight') or k == 'action_proj.weight':
                try:
                    action_emb_dim = int(v.shape[1])
                except Exception:
                    action_emb_dim = None
                break
        if action_emb_dim is None:
            action_emb_dim = args.num_actions
    # determine num_actions from checkpoint action_list if available
    if ckpt_action_list is not None:
        num_actions = len(ckpt_action_list)
    else:
        # fall back to args.num_actions or number of unique dataset actions
        num_actions = args.num_actions if args.num_actions is not None else len(all_actions)

    # Build action embedding compatible with the decoder's expected action_emb_dim
    if action_embedding is not None:
        # Ensure action_idx is within range
        action_idx = int(action_idx) % num_actions
        action_emb_vec = action_embedding(torch.tensor([action_idx], device=device))
        action_emb_vec = action_emb_vec.squeeze(0)
    else:
        # Fallback: create a one-hot like vector of length action_emb_dim
        action_onehot = torch.zeros(action_emb_dim, device=device)
        action_onehot[action_idx % action_emb_dim] = 1.0
        action_emb_vec = action_onehot

    # Instantiate mapping network with matched action_emb_dim
    mapping_net = profile_modules.ProfileActionToMotionTransformer(
        profile_dim=profile_vec.shape[0], codebook_dim=code_embed_dim, seq_len=seq_len, action_emb_dim=action_emb_dim
    ).to(device)

    # Load decoder weights into mapping_net
    mapping_net.load_state_dict(dec_sd, strict=False)
    mapping_net.eval()

    # Run decoder to get predicted latent embeddings / indices
    with torch.no_grad():
        p = profile_vec.unsqueeze(0)
        a = action_emb_vec.unsqueeze(0)
        pred_latents = mapping_net(p, a)  # (1, seq_len, code_embed_dim)
        codebook = model.vqvae.quantizer.codebook.to(device)  # (K, De)
        logits = F.linear(pred_latents, codebook)  # (1, seq_len, K)
        idx = logits.argmax(dim=-1)  # (1, seq_len)
        idx_cpu = idx.squeeze(0).cpu().numpy().astype(np.int64)
        embeddings = codebook[idx.squeeze(0)].cpu().numpy()  # (seq_len, De)

    # Try to decode into motion using VQ-VAE decoder (multiple fallbacks)
    generated_motion = None
    try:
        # Many decoders expect (B, De, seq_len)
        emb_t = torch.tensor(embeddings, device=device, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        if hasattr(model.vqvae, 'decoder'):
            generated_motion = model.vqvae.decoder(emb_t)
            print('Decoded using model.vqvae.decoder with embeddings transposed to (B, De, seq_len)')
    except Exception as e:
        print(f'Decoder attempt 1 failed: {e}')

    if generated_motion is None:
        try:
            emb_t2 = torch.tensor(embeddings, device=device, dtype=torch.float32).unsqueeze(0)
            if hasattr(model.vqvae, 'decoder'):
                generated_motion = model.vqvae.decoder(emb_t2)
                print('Decoded using model.vqvae.decoder with embeddings shape (B, seq_len, De)')
        except Exception as e:
            print(f'Decoder attempt 2 failed: {e}')

    if generated_motion is None:
        # Some VQ-VAE implementations provide a helper to decode from indices
        try:
            if hasattr(model.vqvae, 'decode_from_indices'):
                generated_motion = model.vqvae.decode_from_indices(idx)
                print('Decoded using model.vqvae.decode_from_indices(indices)')
        except Exception as e:
            print(f'Decoder attempt 3 failed: {e}')

    # If we have motion tensor, convert to numpy and save; otherwise save indices and embeddings
    out_file = args.out_file
    if generated_motion is not None:
        try:
            gm = generated_motion.detach().cpu().numpy()
        except Exception:
            gm = np.array(generated_motion)
        np.save(out_file, gm)
        print(f'Generated motion saved to {out_file} (shape {gm.shape})')
    else:
        fallback = out_file if out_file.endswith('.npy') else out_file + '.npy'
        np.save(fallback, {'indices': idx_cpu, 'embeddings': embeddings})
        print(f'Could not decode to joint-space motion; saved indices+embeddings to {fallback}')


if __name__ == '__main__':
    main()
