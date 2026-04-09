import argparse
import os
import csv
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from tqdm import tqdm
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import SpeechSeparator
from dataset import LibriSpeechMixDataset

def main(args):
    # Initialize accelerate
    # Passing mixed_precision='bf16' below activates H200 BFloat16 optimization natively 
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.log_dir, mixed_precision=args.mixed_precision)
    accelerator.init_trackers("conv_tasnet_libri")
    
    # We parse multiple data paths separated by comma if user provides multiple (e.g. train-clean-100,train-clean-360)
    train_dirs = [d.strip() for d in args.train_dirs.split(',')]
    val_dirs = [d.strip() for d in args.val_dirs.split(',')]
    
    # Isolate checkpoints via experiment name so old models aren't overwritten
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize structured CSV log file
    log_csv_path = os.path.join(args.save_dir, "training_log.csv")
    if accelerator.is_local_main_process:
        # Only write header if file doesn't exist (supports resume)
        if not os.path.exists(log_csv_path):
            with open(log_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "train_si_sdr", "val_si_sdr", "learning_rate", "best_model_saved"])
            print(f"[Logger] Created training log: {log_csv_path}")
        else:
            print(f"[Logger] Appending to existing training log: {log_csv_path}")
    
    # Dataset and Dataloader
    # Training set gets noise injection if noise_dir is specified; validation stays clean for fair SI-SDR measurement
    train_dataset = LibriSpeechMixDataset(train_dirs, num_speakers=args.n_src, sample_rate=args.sample_rate, 
                                          epoch_length=args.train_steps*args.batch_size,
                                          noise_dir=args.noise_dir, noise_snr_low=args.noise_snr_low, noise_snr_high=args.noise_snr_high)
    val_dataset = LibriSpeechMixDataset(val_dirs, num_speakers=args.n_src, sample_rate=args.sample_rate, epoch_length=args.val_steps*args.batch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
        
    # Model
    model = SpeechSeparator(n_src=args.n_src, sample_rate=args.sample_rate, 
                            n_blocks=args.n_blocks, n_repeats=args.n_repeats, 
                            bn_chan=args.bn_chan, hid_chan=args.hid_chan)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Loss Function
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    # Prepare Everything
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    start_epoch = 0
    if hasattr(args, 'resume_from') and args.resume_from is not None and os.path.exists(args.resume_from):
        if accelerator.is_local_main_process:
            print(f"Resuming training from {args.resume_from}...")
        # Unwrap model to load state dict if it was wrapped by accelerate
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = torch.load(args.resume_from, map_location="cpu")
        unwrapped_model.load_state_dict(state_dict)
        
        # Attempt to reliably parse the epoch from the checkpoint name (e.g., checkpoint_epoch_70.pt)
        basename = os.path.basename(args.resume_from)
        if "epoch_" in basename:
            try:
                start_epoch = int(basename.split("epoch_")[1].split(".")[0])
                if accelerator.is_local_main_process:
                    print(f"Successfully fast-forwarded to Epoch {start_epoch + 1}")
            except Exception:
                pass
    
    best_val_loss = float('inf')
    
    # Training Loop Outline
    for epoch in range(start_epoch, args.epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        if accelerator.is_local_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        else:
            pbar = train_loader
            
        for mix, targets in pbar:
            estimates = model(mix)
            loss = loss_func(estimates, targets)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            train_loss += loss.item()
            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_train_loss = accelerator.gather(torch.tensor(train_loss / len(train_loader)).to(accelerator.device)).mean().item()
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if accelerator.is_local_main_process:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            else:
                val_pbar = val_loader
                
            for mix, targets in val_pbar:
                estimates = model(mix)
                loss = loss_func(estimates, targets)
                val_loss += loss.item()
                
        avg_val_loss = accelerator.gather(torch.tensor(val_loss / len(val_loader)).to(accelerator.device)).mean().item()
        
        # Logging & Checkpointing on Main Rank
        if accelerator.is_local_main_process:
            train_sisdr = -avg_train_loss
            val_sisdr = -avg_val_loss
            current_lr = optimizer.param_groups[0]['lr']
            saved_best = False
            
            print(f"Epoch {epoch+1} Summary: Train SI-SDR={train_sisdr:.4f} | Val SI-SDR={val_sisdr:.4f} | LR={current_lr:.6f}")
            accelerator.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr}, step=epoch)
            
            # Save the Best Model
            os.makedirs(args.save_dir, exist_ok=True)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save(accelerator.unwrap_model(model).state_dict(), best_path)
                print(f"--> Saved new Best Model! (Loss improved to {best_val_loss:.4f})")
                saved_best = True
                
            # Optional periodic checkpoint
            if (epoch + 1) % args.save_freq == 0:
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            
            # Append row to structured CSV log
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    f"{train_sisdr:.4f}",
                    f"{val_sisdr:.4f}",
                    f"{current_lr:.6f}",
                    "YES" if saved_best else ""
                ])
                
        # Update Scheduler (needs to step on all processes)
        scheduler.step(avg_val_loss)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Distributed Multi-GPU Training for Conv-TasNet on LibriSpeech")
    # Multiple directories can be comma-separated
    parser.add_argument("--train_dirs", type=str, required=True, help="Path to train subsets (e.g. data/train-clean-100,data/train-clean-360)")
    parser.add_argument("--val_dirs", type=str, required=True, help="Path to validation subset (e.g. data/dev-clean)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for saving unique checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Tensorboard log directory")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision mode (use bf16 for H200)")
    
    parser.add_argument("--n_src", type=int, default=2, help="Number of speakers")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Sample rate in Hz")
    parser.add_argument("--train_steps", type=int, default=1000, help="Number of virtual train batches per epoch")
    parser.add_argument("--val_steps", type=int, default=200, help="Number of virtual validation batches per epoch")
    
    # HP Tuning Arch
    parser.add_argument("--n_blocks", type=int, default=8, help="Conv-TasNet X (Blocks)")
    parser.add_argument("--n_repeats", type=int, default=3, help="Conv-TasNet R (Repeats)")
    parser.add_argument("--bn_chan", type=int, default=128, help="Conv-TasNet B (Bottleneck channels)")
    parser.add_argument("--hid_chan", type=int, default=512, help="Conv-TasNet H (Hidden channels)")
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size PER GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--save_freq", type=int, default=5)
    
    parser.add_argument("--resume_from", type=str, default=None, help="Absolute or relative path to a .pt checkpoint to resume training from")
    
    # Noise augmentation (V5+)
    parser.add_argument("--noise_dir", type=str, default=None, help="Path to MUSAN noise directory for noise-resilient training")
    parser.add_argument("--noise_snr_low", type=float, default=5.0, help="Minimum SNR (dB) for noise injection (lower = more noise)")
    parser.add_argument("--noise_snr_high", type=float, default=20.0, help="Maximum SNR (dB) for noise injection (higher = less noise)")
    
    args = parser.parse_args()
    main(args)
