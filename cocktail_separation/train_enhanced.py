#!/usr/bin/env python3
"""
Enhanced DPRNN Training Script with Progress Tracking, ETA Estimation, and GPU Management
Uses GPUs 5 and 6 with real-time monitoring and file-based result tracking
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import subprocess
import json

def get_gpu_memory_available(gpu_id: int) -> float:
    """Get available GPU memory in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=memory.free", f"--id={gpu_id}", "--format=csv,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split('\n')[-1]) / 1024
    except:
        return 0.0

def create_training_status_file(phase: int, num_speakers: int, output_dir: Path) -> Path:
    """Create a training status file for tracking progress"""
    status_file = output_dir / f"training_status_phase_{phase}_{num_speakers}spk.json"
    
    initial_status = {
        "phase": phase,
        "num_speakers": num_speakers,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "status": "RUNNING",
        "gpus": [5, 6],
        "total_epochs": None,
        "current_epoch": 0,
        "best_val_sisnr": -float('inf'),
        "best_val_sisnr_epoch": -1,
        "training_history": {
            "epoch": [],
            "train_loss": [],
            "train_sisnr": [],
            "val_loss": [],
            "val_sisnr": [],
            "learning_rate": [],
            "grad_norm": [],
            "epoch_time": []
        },
        "time_per_epoch_estimates": [],
        "eta_hours": None,
    }
    
    with open(status_file, 'w') as f:
        json.dump(initial_status, f, indent=2)
    
    return status_file

def update_training_status(
    status_file: Path,
    epoch: int,
    train_loss: float,
    train_sisnr: float,
    val_loss: Optional[float] = None,
    val_sisnr: Optional[float] = None,
    learning_rate: float = 0.0,
    grad_norm: float = 0.0,
    epoch_time: float = 0.0,
    total_epochs: Optional[int] = None
):
    """Update training status JSON with current metrics"""
    
    with open(status_file) as f:
        status = json.load(f)
    
    status["current_epoch"] = epoch
    
    if total_epochs and status["total_epochs"] is None:
        status["total_epochs"] = total_epochs
    
    status["training_history"]["epoch"].append(epoch)
    status["training_history"]["train_loss"].append(train_loss)
    status["training_history"]["train_sisnr"].append(train_sisnr)
    status["training_history"]["learning_rate"].append(learning_rate)
    status["training_history"]["grad_norm"].append(grad_norm)
    status["training_history"]["epoch_time"].append(epoch_time)
    
    if val_loss is not None:
        status["training_history"]["val_loss"].append(val_loss)
    if val_sisnr is not None:
        status["training_history"]["val_sisnr"].append(val_sisnr)
        if val_sisnr > status["best_val_sisnr"]:
            status["best_val_sisnr"] = val_sisnr
            status["best_val_sisnr_epoch"] = epoch
    
    # Calculate ETA
    if len(status["training_history"]["epoch_time"]) > 1:
        avg_epoch_time = sum(status["training_history"]["epoch_time"]) / len(status["training_history"]["epoch_time"])
        if status["total_epochs"]:
            remaining_epochs = status["total_epochs"] - epoch
            eta_seconds = remaining_epochs * avg_epoch_time
            status["eta_hours"] = round(eta_seconds / 3600, 2)
            status["time_per_epoch_estimates"].append({
                "epoch": epoch,
                "avg_time_per_epoch": round(avg_epoch_time, 2),
                "eta_hours": status["eta_hours"]
            })
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def check_gpu_availability(gpu_ids: list = [5, 6], min_memory_gb: float = 130):
    """Check if GPUs are available with sufficient memory"""
    print(f"\n{'='*60}")
    print(f"GPU Availability Check")
    print(f"{'='*60}")
    
    all_available = True
    for gpu_id in gpu_ids:
        avail_mem = get_gpu_memory_available(gpu_id)
        status = "✓" if avail_mem >= min_memory_gb else "✗"
        print(f"GPU {gpu_id}: {avail_mem:.1f} GB available {status}")
        if avail_mem < min_memory_gb:
            all_available = False
    
    if not all_available:
        print(f"\nWARNING: Some GPUs don't have {min_memory_gb}GB available!")
        print("Waiting 10 seconds before proceeding...")
        time.sleep(10)
    
    print(f"{'='*60}\n")
    return all_available

def launch_training(
    config_path: str,
    phase: int,
    num_speakers: int,
    gpu_ids: list = [5, 6],
    resume_from: Optional[str] = None,
    output_base_dir: Optional[str] = None
):
    """Launch training with GPU management and progress tracking"""
    
    # Check GPU availability
    check_gpu_availability(gpu_ids)
    
    # Setup output directory
    if output_base_dir is None:
        output_base_dir = "./training_results"
    
    output_dir = Path(output_base_dir) / f"phase_{phase}_{num_speakers}spk"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create status tracking file
    status_file = create_training_status_file(phase, num_speakers, output_dir)
    print(f"\n📊 Training Status File: {status_file}")
    print(f"📁 Results Directory: {output_dir}")
    
    # Set environment variables for GPU selection
    gpu_env = os.environ.copy()
    gpu_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Prepare training command
    cmd = [
        sys.executable,
        "train.py",
        "--config", config_path,
    ]
    
    if resume_from:
        cmd.extend(["--resume", resume_from])
    
    # Add status file path for logging
    cmd.extend(["--status-file", str(status_file)])
    cmd.extend(["--output-dir", str(output_dir)])
    
    print(f"\n🚀 Launching training with GPUs: {gpu_ids}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print(f"Speakers: {num_speakers}")
    print(f"{'='*60}\n")
    
    # Launch training
    print("⏱️  Training in progress. Real-time progress and ETA shown above.")
    print("Monitor progress at: cat " + str(status_file))
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, env=gpu_env, cwd="/mnt/raid/rl_gaming/RL4VLM2/cocktail_separation")
        
        # Update final status
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        total_time = time.time() - start_time
        status["end_time"] = datetime.now().isoformat()
        status["status"] = "COMPLETED" if result.returncode == 0 else "FAILED"
        status["total_training_time_hours"] = round(total_time / 3600, 2)
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Finished: {status['status']}")
        print(f"Total Time: {status['total_training_time_hours']} hours")
        print(f"Best Validation SI-SNR: {status['best_val_sisnr']:.2f} dB (Epoch {status['best_val_sisnr_epoch']})")
        print(f"{'='*60}\n")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
        with open(status_file, 'r') as f:
            status = json.load(f)
        status["status"] = "INTERRUPTED"
        status["end_time"] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        return 1

def run_full_curriculum():
    """Run the full 2→3→4→5 speaker curriculum"""
    
    print("\n" + "="*70)
    print("     DPRNN-TasNet Full Curriculum Training (2→3→4→5 speakers)")
    print("="*70 + "\n")
    
    configs = [
        (1, 2, "configs/2spk.yaml"),
        (2, 3, "configs/3spk.yaml"),
        (3, 4, "configs/4spk.yaml"),
        (4, 5, "configs/5spk.yaml"),
    ]
    
    output_base_dir = "./training_results"
    resume_from = None
    
    for phase, num_speakers, config_path in configs:
        print(f"\n{'#'*70}")
        print(f"# PHASE {phase}: {num_speakers}-speaker separation")
        print(f"{'#'*70}\n")
        
        # Check if checkpoint from previous phase exists
        if phase > 1:
            prev_phase = phase - 1
            prev_checkpoint = Path(output_base_dir) / f"phase_{prev_phase}_{num_speakers-1}spk" / "best.pt"
            if prev_checkpoint.exists():
                resume_from = str(prev_checkpoint)
                print(f"📦 Resuming from: {resume_from}\n")
        
        exit_code = launch_training(
            config_path,
            phase,
            num_speakers,
            gpu_ids=[5, 6],
            resume_from=resume_from,
            output_base_dir=output_base_dir
        )
        
        if exit_code != 0:
            print(f"\n❌ Phase {phase} failed. Aborting curriculum.")
            return exit_code
        
        # Update for next phase
        resume_from = None
        print(f"\n✅ Phase {phase} completed successfully!\n")
    
    print("\n" + "="*70)
    print("    ✨ Full Curriculum Training Completed Successfully! ✨")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced DPRNN Training with GPU Management")
    parser.add_argument("--phase", type=int, default=None, help="Specific phase to run (1-4)")
    parser.add_argument("--full-curriculum", action="store_true", help="Run full 2→3→4→5 curriculum")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[5, 6], help="GPU IDs to use")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="./training_results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.full_curriculum:
        exit_code = run_full_curriculum()
    elif args.phase and args.config:
        num_speakers = args.phase + 1
        exit_code = launch_training(
            args.config,
            args.phase,
            num_speakers,
            gpu_ids=args.gpu_ids,
            resume_from=args.resume,
            output_base_dir=args.output_dir
        )
    else:
        parser.print_help()
        exit_code = 0
    
    sys.exit(exit_code)
