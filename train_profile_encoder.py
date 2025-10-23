import matplotlib.pyplot as plt
import numpy as np
import options.option_profile_encoder as option_profile_encoder
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import umap
import utils.utils_model as utils_model
import warnings
from collections import defaultdict
from dataset import dataset_183_retarget
from models import profile_modules
from models import vqvae
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

def _use_arg(args, name, default):
    """Return args.name if args is provided and has the attribute, else default."""
    if args is None:
        return default
    return getattr(args, name) if hasattr(args, name) else default

def get_biomech_vector(subject, dataset, biomech_dim, args=None):
    biomech_dim = _use_arg(args, 'biomech_dim', biomech_dim)
    bm = dataset.get_biomech(subject)
    if isinstance(bm, dict):
        vec = bm.get('vector', None)
        if vec is None:
            vec = dataset._biomech_to_vector(
                bm,
                target_samples=dataset.default_biomech_target_samples,
                biomech_dim=dataset.default_biomech_dim
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
sport_list = set()
level_list = set()
injury_types = set()

def get_metadata_vector(subject):
    """Create metadata one-hot encodings for ProfileEncoder from subject metadata"""
    metadata = loader.dataset.subject_metadata.get(subject, {})

    # Encode categorical features as one-hot
    age = metadata.get('age', 'unknown')
    height = metadata.get('height', 'unknown')
    mass = metadata.get('mass', 'unknown')

    sex = metadata.get('sex', 'unknown')
    sport = metadata.get('sport', 'unknown')
    level = metadata.get('level', 'unknown')
    injuries = metadata.get('injuries', [])
    
    # Sex encoding (one-hot: male, female, unknown)
    sex_encoding = [1.0, 0.0, 0.0] if sex == 'male' else [0.0, 1.0, 0.0] if sex == 'female' else [0.0, 0.0, 1.0]
    
    # Sport encoding (one-hot for common sports)

    sport_encoding = [1.0 if sport == s else 0.0 for s in sport_list]

    level_encoding = [1.0 if level == l else 0.0 for l in level_list]
    

    
    # Filter out NaN values and ensure injuries is a list of strings
    if injuries is None or (isinstance(injuries, float) and np.isnan(injuries)):
        injuries = []
    elif not isinstance(injuries, list):
        injuries = []
    
    # Convert all injury entries to strings and filter out NaN values
 
    injury_encoding = [0.0] * len(injury_types)
    for injury in injuries:
        if injury is not None and not (isinstance(injury, float) and np.isnan(injury)):
            injury_encoding[inj_idx[injury]] += 1.0
            
    
    # Combine all one-hot encodings
    metadata_onehot = np.array([
        age,
        height,
        mass,
        *sex_encoding,
        *sport_encoding, 
        *level_encoding,
        *injury_encoding
    ], dtype=np.float32)
    
    return metadata_onehot


def sample_triplet(subjects, subject_latents, biomech_dim, device):
    subject_pos = np.random.choice(subjects)
    latents_pos = subject_latents[subject_pos]
    if len(latents_pos) < 2:
        return None
    idx = np.random.choice(len(latents_pos), 2, replace=False)
    anchor = latents_pos[idx[0]].unsqueeze(0).to(device)
    positive = latents_pos[idx[1]].unsqueeze(0).to(device)
    negative_subjects = [s for s in subjects if s != subject_pos and len(subject_latents[s]) > 0]
    if not negative_subjects:
        return None
    subject_neg = np.random.choice(negative_subjects)
    latents_neg = subject_latents[subject_neg]
    negative = latents_neg[np.random.choice(len(latents_neg))].unsqueeze(0).to(device)
    
    # FIXED: Return consistent number of values - only metadata indices for ProfileEncoder
    a_metadata = get_metadata_vector(subject_pos)
    p_metadata = a_metadata.copy()
    n_metadata = get_metadata_vector(subject_neg)

    a_bio = get_biomech_vector(subject_pos)
    p_bio = a_bio.copy()
    n_bio = get_biomech_vector(subject_neg)
    
    return anchor, positive, negative, a_metadata, p_metadata, n_metadata, a_bio, p_bio, n_bio

def create_batch(batch_size, subjects, subject_latents, biomech_dim, device):
    anchors, positives, negatives = [], [], []
    anchor_metadatas, positive_metadatas, negative_metadatas = [], [], []
    anchor_bios, positive_bios, negative_bios = [], [], []
    for _ in range(batch_size):
        result = sample_triplet(subjects, subject_latents, biomech_dim, device)
        if result is None:
            continue
        anchor, positive, negative, a_metadata, p_metadata, n_metadata, a_bio, p_bio, n_bio = result
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        anchor_metadatas.append(a_metadata)
        positive_metadatas.append(p_metadata)
        negative_metadatas.append(n_metadata)
        anchor_bios.append(a_bio)
        positive_bios.append(p_bio)
        negative_bios.append(n_bio)
    return anchors, positives, negatives, anchor_metadatas, positive_metadatas, negative_metadatas, anchor_bios, positive_bios, negative_bios

def compute_triplet_accuracy(anchor_emb, positive_emb, negative_emb):
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
    neg_sim = F.cosine_similarity(anchor_emb, negative_emb)
    return (pos_sim > neg_sim).float().mean().item()

def evaluate(encoder, subjects, subject_samples, biomech_dim, device, dataset, batch_size, loss_fn, args=None):
    batch_size = _use_arg(args, 'batch_size', batch_size)
    biomech_dim = _use_arg(args, 'biomech_dim', biomech_dim)
    device = _use_arg(args, 'device', device)
    encoder.eval()
    anchors, positives, negatives, anchor_metadatas, positive_metadatas, negative_metadatas, anchor_bios, positive_bios, negative_bios = create_batch(
        batch_size, subjects, subject_samples, biomech_dim, device, dataset, args=args
    )
    if len(anchors) == 0:
        return None, None
    anchor_batch = torch.cat(anchors, dim=0)
    positive_batch = torch.cat(positives, dim=0)
    negative_batch = torch.cat(negatives, dim=0)
    anchor_bio_batch = torch.tensor(np.stack(anchor_bios), device=device, dtype=torch.float32)
    positive_bio_batch = torch.tensor(np.stack(positive_bios), device=device, dtype=torch.float32)
    negative_bio_batch = torch.tensor(np.stack(negative_bios), device=device, dtype=torch.float32)
    anchor_metadata_batch = torch.tensor(np.stack(anchor_metadatas), device=device, dtype=torch.float32)
    positive_metadata_batch = torch.tensor(np.stack(positive_metadatas), device=device, dtype=torch.float32)
    negative_metadata_batch = torch.tensor(np.stack(negative_metadatas), device=device, dtype=torch.float32)
    with torch.no_grad():
        anchor_emb = F.normalize(encoder(anchor_batch, anchor_bio_batch), dim=1)
        positive_emb = F.normalize(encoder(positive_batch, positive_bio_batch), dim=1)
        negative_emb = F.normalize(encoder(negative_batch, negative_bio_batch), dim=1)
        loss = loss_fn(anchor_emb, positive_emb, negative_emb).item()
        acc = compute_triplet_accuracy(anchor_emb, positive_emb, negative_emb)
    return loss, acc

def get_embeddings(encoder, subjects, subject_samples, biomech_dim, device, dataset, batch_size, n_batches=10, args=None):
    batch_size = _use_arg(args, 'batch_size', batch_size)
    biomech_dim = _use_arg(args, 'biomech_dim', biomech_dim)
    device = _use_arg(args, 'device', device)
    encoder.eval()
    all_embs = []
    all_labels = []
    for _ in range(n_batches):
        anchors, positives, negatives, anchor_bios, positive_bios, negative_bios = create_batch(
            batch_size, subjects, subject_samples, biomech_dim, device, dataset, args=args
        )
        if len(anchors) == 0:
            continue
        anchor_batch = torch.cat(anchors, dim=0)
        anchor_bio_batch = torch.tensor(np.stack(anchor_bios), device=device, dtype=torch.float32)
        with torch.no_grad():
            emb = F.normalize(encoder(anchor_batch, anchor_bio_batch), dim=1).cpu().numpy()
        all_embs.append(emb)
        # For each anchor in this batch, find its subject label
        for anchor in anchors:
            found = False
            for subj in subjects:
                for latent, _ in subject_samples[subj]:
                    if torch.equal(latent, anchor.squeeze(0).cpu()):
                        all_labels.append(subj)
                        found = True
                        break
                if found:
                    break
            if not found:
                all_labels.append("unknown")
    if all_embs:
        return np.concatenate(all_embs, axis=0), np.array(all_labels)
    else:
        return np.array([]), np.array([])

def plot_tsne(embs, labels, save_path, title):
    if embs.shape[0] < 2:
        print(f"Not enough embeddings for t-SNE plot: {title}")
        return
    tsne = TSNE(n_components=2, random_state=42)
    embs_2d = tsne.fit_transform(embs)
    _, label_indices = np.unique(labels, return_inverse=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=label_indices, cmap='tab20', s=10, alpha=0.7)
    plt.title(title + " (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_umap(embs, labels, save_path, title):
    if embs.shape[0] < 2:
        print(f"Not enough embeddings for UMAP plot: {title}")
        return
    reducer = umap.UMAP(n_components=2, random_state=42)
    embs_2d = reducer.fit_transform(embs)
    _, label_indices = np.unique(labels, return_inverse=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=label_indices, cmap='tab20', s=10, alpha=0.7)
    plt.title(title + " (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pca(embs, labels, save_path, title):
    if embs.shape[0] < 2:
        print(f"Not enough embeddings for PCA plot: {title}")
        return
    reducer = PCA(n_components=2)
    embs_2d = reducer.fit_transform(embs)
    _, label_indices = np.unique(labels, return_inverse=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=label_indices, cmap='tab20', s=10, alpha=0.7)
    plt.title(title + " (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = option_profile_encoder.get_args_parser()

    # Set global seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = utils_model.get_logger(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vqvae.HumanVQVAE(
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

    model.to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    # Only load weights, not args or other metadata
    state_dict = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    loader = dataset_183_retarget.retargeted183_data_loader(data_dir=args.dataset_path, num_workers=4, pre_load=True)

    encoder = model.vqvae.encoder
    for value in loader.dataset.subject_metadata.values():
        sport_list.add(value['sport'])
        level_list.add(value['level'])
        injury_types.update(value['injuries'])
    sport_list = sorted(list(sport_list))
    level_list = sorted(list(level_list))
    # Filter out NaN values and keep only valid injury types
    injury_types = sorted([a for a in injury_types if a is not None and not (isinstance(a, float) and np.isnan(a))])
    inj_idx = {a:i for i,a in enumerate(injury_types)}
    subject_samples = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding Motions")):
            try:
                motions, _, names, subject_names = batch
            except ValueError as e:
                logger.info(f"Skipping batch due to error: {e}")
                continue
            except StopIteration:
                break
            motions = motions.to(device, dtype=torch.float32)
            latents = model.vqvae.encoder(model.vqvae.preprocess(motions))
            for latent, subject_name, action_name in zip(latents, subject_names, names):
                subject_samples[subject_name].append((latent.cpu(), action_name))

    rng = np.random.RandomState(args.seed)
    _all_subjects = sorted([s for s in subject_samples.keys() if len(subject_samples[s]) > 0])
    n_test = 33
    n_train_target = 150
    if len(_all_subjects) < 1:
        raise RuntimeError("No subjects with motions found in subject_samples.")
    if len(_all_subjects) <= n_test:
        raise RuntimeError(f"Not enough subjects ({len(_all_subjects)}) to reserve {n_test} for validation.")

    test_subjects = list(rng.choice(_all_subjects, size=min(n_test, len(_all_subjects)), replace=False))
    _remaining = [s for s in _all_subjects if s not in test_subjects]
    train_candidates = [s for s in _remaining if len(subject_samples[s]) >= 2]

    if len(train_candidates) < n_train_target:
        warnings.warn(f"Only {len(train_candidates)} subjects have >=2 motions; will use all of them for training.")
        train_subjects = train_candidates
    else:
        rng.shuffle(train_candidates)
        train_subjects = train_candidates[:n_train_target]

    subjects = list(train_subjects)
    logger.info(f"Using {len(subjects)} subjects for training, reserving {len(test_subjects)} subjects for validation.")

    latent_dim = args.latent_dim
    profile_dim = args.profile_dim
    bio_out_dim = args.bio_out_dim
    dataset = loader.dataset
    biomech_dim = getattr(dataset, "default_biomech_dim", args.biomech_dim)

    num_epochs = args.total_epochs
    batch_size = args.batch_size

    encoder = profile_modules.ProfileEncoder(
        latent_dim=latent_dim,
        profile_dim=profile_dim,
        bio_out_dim=bio_out_dim,
        biomech_dim=biomech_dim
    ).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = args.warmup_epochs
    main_epochs = max(1, num_epochs * 2 - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)
        ],
        milestones=[warmup_epochs],
    )

    loss_fn = nn.TripletMarginLoss()
    all_losses = []
    val_losses = []
    val_accs = []
    val_epochs = []
    for epoch in range(num_epochs):
        if len(subjects) == 0:
            raise RuntimeError("No subjects with >=2 motions found.")
        np.random.shuffle(subjects)

        anchors, positives, negatives, anchor_bios, positive_bios, negative_bios = create_batch(
            batch_size, subjects, subject_samples, biomech_dim, device, dataset, args=args
        )
        if len(anchors) == 0:
            continue

        anchor_batch = torch.cat(anchors, dim=0)
        positive_batch = torch.cat(positives, dim=0)
        negative_batch = torch.cat(negatives, dim=0)
        anchor_bio_batch = torch.tensor(np.stack(anchor_bios), device=device, dtype=torch.float32)
        positive_bio_batch = torch.tensor(np.stack(positive_bios), device=device, dtype=torch.float32)
        negative_bio_batch = torch.tensor(np.stack(negative_bios), device=device, dtype=torch.float32)
        anchor_metadata_batch = torch.tensor(np.stack(anchor_metadatas), device=device, dtype=torch.float32)
        positive_metadata_batch = torch.tensor(np.stack(positive_metadatas), device=device, dtype=torch.float32)
        negative_metadata_batch = torch.tensor(np.stack(negative_metadatas), device=device, dtype=torch.float32)

        encoder.train()
        anchor_emb = F.normalize(encoder(anchor_batch, anchor_bio_batch, anchor_metadata_batch), dim=1)
        positive_emb = F.normalize(encoder(positive_batch, positive_bio_batch, positive_metadata_batch), dim=1)
        negative_emb = F.normalize(encoder(negative_batch, negative_bio_batch, negative_metadata_batch), dim=1)

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        all_losses.append(loss.item())

        val_loss_str = ""
        val_acc_str = ""
        # Validation every 100 epochs or last epoch
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            val_loss, val_acc = evaluate(
                encoder, test_subjects, subject_samples, biomech_dim, device, dataset, batch_size, loss_fn, args=args
            )
            if val_loss is not None:
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_epochs.append(epoch + 1)
                val_loss_str = f" | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.3f}"
            else:
                val_loss_str = " | Not enough validation samples."

        if (epoch + 1) % 100 == 0 or (val_loss_str and (epoch + 1) % 100 == 0) or (epoch == num_epochs - 1):
            msg = f"Encoder Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.6f}{val_loss_str}"
            logger.info(msg)

    logger.info("Training complete.")

    # Plot and save training loss curve
    plt.figure()
    plt.plot(range(1, len(all_losses) + 1), all_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    train_loss_plot_path = os.path.join(save_dir, "training_loss.pdf")
    plt.savefig(train_loss_plot_path)
    plt.close()
    logger.info(f"Training loss curve saved to {train_loss_plot_path}")

    # Plot and save validation loss curve
    if val_losses:
        plt.figure()
        plt.plot(val_epochs, val_losses, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.tight_layout()
        val_loss_plot_path = os.path.join(save_dir, "validation_loss.pdf")
        plt.savefig(val_loss_plot_path)
        plt.close()
        logger.info(f"Validation loss curve saved to {val_loss_plot_path}")

    encoder_save_path = os.path.join(save_dir, "profile_encoder.pth")
    # Save both the encoder weights and the args used to construct/training
    torch.save({'net': encoder.state_dict(), 'args': vars(args)}, encoder_save_path)
    encoder.eval()
    logger.info(f"Encoder weights saved to {encoder_save_path}")

    # After training loop and loss plots
    train_embs, train_labels = get_embeddings(
        encoder, subjects, subject_samples, biomech_dim, device, dataset, batch_size=128, n_batches=10, args=args
    )
    val_embs, val_labels = get_embeddings(
        encoder, test_subjects, subject_samples, biomech_dim, device, dataset, batch_size=128, n_batches=10, args=args
    )

    # Save embedding plots
    plot_tsne(train_embs, train_labels, os.path.join(save_dir, "train_embeddings_tsne.pdf"), "Training Embeddings")
    plot_tsne(val_embs, val_labels, os.path.join(save_dir, "validation_embeddings_tsne.pdf"), "Validation Embeddings")
    plot_umap(train_embs, train_labels, os.path.join(save_dir, "train_embeddings_umap.pdf"), "Training Embeddings")
    plot_umap(val_embs, val_labels, os.path.join(save_dir, "validation_embeddings_umap.pdf"), "Validation Embeddings")
    plot_pca(train_embs, train_labels, os.path.join(save_dir, "train_embeddings_pca.pdf"), "Training Embeddings")
    plot_pca(val_embs, val_labels, os.path.join(save_dir, "validation_embeddings_pca.pdf"), "Validation Embeddings")
    logger.info(f"Embedding plots saved to {save_dir}")

if __name__ == "__main__":
    main()