import deepspeed
import json
import models.vqvae as vqvae
import nimblephysics as nimble
import options.option_vqvae as option_vqvae
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import utils.eval_trans as eval_trans
import utils.losses as losses 
import utils.utils_model as utils_model
import warnings
from dataset import dataset_183_retarget
from dataset import dataset_addbiomechanics
from models.evaluator_wrapper import EvaluatorModelWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.word_vectorizer import WordVectorizer
warnings.filterwarnings('ignore')

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


def main():
    # --- Robust device and distributed/deepspeed setup ---
    args = option_vqvae.get_args_parser()
    torch.manual_seed(args.seed)

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        if dist.is_available() and not dist.is_initialized() and world_size > 1:
            dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cpu")

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)

    # Logger
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.dataname == '183_athletes':
        args.nb_joints = 37
        train_loader = dataset_183_retarget.retargeted183_data_loader(
            window_size=args.window_size,
            unit_length=2**args.down_t,
            batch_size=args.batch_size,
            num_workers=4,
            data_dir='/home/mnt/datasets/183_retargeted',
            pre_load=True
        )
        train_loader_iter = dataset_183_retarget.cycle(train_loader)
    elif args.dataname == 'addbiomechanics':
        args.nb_joints = 23
        train_loader = dataset_addbiomechanics.addb_data_loader(
            window_size=args.window_size,
            unit_length=2**args.down_t,
            batch_size=args.batch_size,
            mode='train',
            data_dir='/home/mnt/AddBiomechanics'
        )
    else:
        logger.error(f"Invalid dataset name: {args.dataname}")
        raise ValueError(f"Dataset '{args.dataname}' is not supported.")

    logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

    # Setup VQ-VAE model
    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm
    )

    if args.resume_pth : 
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location=device)
        # Accept either a raw state_dict or a wrapped checkpoint containing the state dict under 'net'
        sd = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
        # If saved from DataParallel the keys may be prefixed with 'module.' — strip that
        if isinstance(sd, dict):
            new_sd = {}
            for k, v in sd.items():
                new_k = k[len('module.'):] if k.startswith('module.') else k
                new_sd[new_k] = v
            sd = new_sd
        # Load only the weight tensors (state_dict)
        net.load_state_dict(sd, strict=True)
    net.train()
    net.to(device)

    # Optimizer and scheduler setup
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # Run deepspeed if avaliable (2+ GPUs)
    if torch.cuda.is_available() and world_size > 1:
        deepspeed_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "betas": [0.9, 0.99],
                    "weight_decay": args.weight_decay
                }
            },
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0}
        }
        net, optimizer, _, _ = deepspeed.initialize(
            model=net,
            optimizer=optimizer,
            args=args,
            config_params=deepspeed_config
        )
    else:
        logger.info("Running without DeepSpeed (single GPU or CPU).")

    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

    # Warm up
    avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.
    for nb_iter in range(1, args.warm_up_iter):
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        gt_motion, len_motion, _, _ = next(train_loader_iter)
        gt_motion = gt_motion.to(device).float() 

        pred_motion, loss_commit, perplexity = net(gt_motion)
        if not torch.isfinite(pred_motion).all():
            logger.error(f"NaN/Inf in pred_motion at iter {nb_iter}")
            logger.error(f"pred_motion stats: min={pred_motion.min().item()}, max={pred_motion.max().item()}, mean={pred_motion.mean().item()}")
            continue

        loss_motion = Loss(pred_motion, gt_motion)
        loss_temp = torch.mean((pred_motion[:,1:,:] - pred_motion[:,:-1,:])**2)
        if not (torch.isfinite(loss_motion) and torch.isfinite(loss_commit) and torch.isfinite(loss_temp)):
            logger.error(f"NaN/Inf in loss components at iter {nb_iter}: "
                        f"loss_motion={loss_motion.item()}, "
                        f"loss_commit={loss_commit.item()}, "
                        f"loss_temp={loss_temp.item()}")
            continue
        loss = loss_motion + args.commit * loss_commit + args.temporal * loss_temp 

        if not torch.isfinite(loss):
            logger.error(f"NaN or Inf detected in loss at iter {nb_iter}")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_temporal += loss_temp.item()
        
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_temporal /= args.print_iter
            
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f}")
            
            avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.

    # Training Loop
    avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.
    torch.save({'net': net.state_dict(), 'args': vars(args)}, os.path.join(args.out_dir, 'warmup.pth'))
    for nb_iter in range(1, args.total_iter + 1):
        gt_motion, len_motion, _, _ = next(train_loader_iter)
        gt_motion = gt_motion.to(device).float() 
        
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_temp = torch.mean((pred_motion[:,1:,:]-pred_motion[:,:-1,:])**2)
        loss = loss_motion + args.commit * loss_commit + args.temporal * loss_temp 
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_temporal += loss_temp.item()
        
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_temporal /= args.print_iter
            
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Temporal. {avg_temporal:.5f}")
            
            avg_recons, avg_perplexity, avg_commit, avg_temporal = 0., 0., 0., 0.

        if nb_iter % (10 * args.eval_iter) == 0:
            torch.save({'net': net.state_dict(), 'args': vars(args)}, os.path.join(args.out_dir, str(nb_iter) + '.pth'))

if __name__ == "__main__":
    main()
