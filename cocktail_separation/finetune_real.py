#!/usr/bin/env python3
"""
3-GPU DDP Fine-tuning Script targeting 3-speaker separation.
Loads the robust Phase 2 checkpoint and uses DistributedDataParallel.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataloader
from src.model import DPRNNTasNet
from src.losses import pit_loss

def main():
    # Initialize DDP
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    num_speakers = 3
    num_epochs = 50
    batch_size = 4  # Per GPU batch size
    learning_rate = 1e-4  # Smaller for fine-tuning
    
    if global_rank == 0:
        print("\n" + "="*70)
        print("PHASE 2 (3-Speaker) - REAL DATA FINE-TUNING (DDP 3 GPUs)")
        print("="*70 + "\n")
    
    # Paths
    train_root = Path("data/mixtures") / f"{num_speakers}spk" / "train"
    eval_root = Path("data/mixtures") / f"{num_speakers}spk" / "val"
    
    if global_rank == 0 and not train_root.exists():
        print(f"Dataset not found at {train_root}. Please ensure data is generated.")
        dist.destroy_process_group()
        return

    # DataLoaders (distributed=True)
    train_loader = build_dataloader(
        split_root=train_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=4,
        distributed=True,
        shuffle=True
    )
    
    eval_loader = build_dataloader(
        split_root=eval_root,
        num_speakers=num_speakers,
        batch_size=batch_size,
        num_workers=4,
        distributed=True,
        shuffle=False
    )
    
    # Model
    model = DPRNNTasNet(
        num_speakers=num_speakers,
        encoder_dim=64,
        encoder_kernel=2,
        encoder_stride=2,
        bottleneck_dim=64,
        chunk_size=100,
        num_dprnn_blocks=6
    ).to(device)
    
    # Load Phase 2 Best Checkpoint
    ckpt_path = Path("checkpoints/best_phase2.pt")
    if ckpt_path.exists():
        if global_rank == 0:
            print(f"Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt['model_state'])
    elif global_rank == 0:
        print("WARNING: Phase 2 checkpoint not found! Starting from scratch.")

    # Wrap DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=(global_rank==0))
    
    best_sisnr = -float('inf')
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        train_loss = 0
        train_sisnr = 0
        train_count = 0
        
        for batch_idx, (mixture, speakers) in enumerate(train_loader):
            mixture = mixture.to(device)
            speakers = speakers.to(device)
            
            optimizer.zero_grad()
            estimates = model(mixture)
            
            min_len = min(estimates.shape[-1], speakers.shape[-1])
            estimates = estimates[..., :min_len]
            speakers = speakers[..., :min_len]
            
            loss, sisnr = pit_loss(estimates, speakers, snr_weight=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item() * mixture.shape[0]
            train_sisnr += sisnr.item() * mixture.shape[0]
            train_count += mixture.shape[0]
            
            if global_rank == 0 and (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1} | Loss: {loss.item():.4f} | SI-SNR: {sisnr.item():.2f} dB", end='\r')
                
        # Aggregate across GPUs
        metrics = torch.tensor([train_loss, train_sisnr, train_count], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_train_loss = metrics[0].item() / metrics[2].item()
        avg_train_sisnr = metrics[1].item() / metrics[2].item()
        
        # Validation
        model.eval()
        val_loss, val_sisnr, val_count = 0, 0, 0
        
        with torch.no_grad():
            for mixture, speakers in eval_loader:
                mixture = mixture.to(device)
                speakers = speakers.to(device)
                estimates = model(mixture)
                
                min_len = min(estimates.shape[-1], speakers.shape[-1])
                estimates = estimates[..., :min_len]
                speakers = speakers[..., :min_len]
                
                loss, sisnr = pit_loss(estimates, speakers, snr_weight=0.1)
                
                val_loss += loss.item() * mixture.shape[0]
                val_sisnr += sisnr.item() * mixture.shape[0]
                val_count += mixture.shape[0]
                
        metrics_val = torch.tensor([val_loss, val_sisnr, val_count], device=device)
        dist.all_reduce(metrics_val, op=dist.ReduceOp.SUM)
        
        avg_val_loss = metrics_val[0].item() / metrics_val[2].item()
        avg_val_sisnr = metrics_val[1].item() / metrics_val[2].item()
        
        scheduler.step(avg_val_sisnr)
        
        if global_rank == 0:
            print(f"✅ Epoch {epoch+1:2d} | Train SI-SNR: {avg_train_sisnr:6.2f} dB | Val SI-SNR: {avg_val_sisnr:6.2f} dB")
            
            if avg_val_sisnr > best_sisnr:
                best_sisnr = avg_val_sisnr
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'val_sisnr': best_sisnr
                }, 'checkpoints/finetuned_3spk.pt')
                
    if global_rank == 0:
        print("\n🎉 Fine-tuning Complete!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
