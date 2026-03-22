#!/usr/bin/env python3
"""
Real-time Training Monitor for DPRNN
Displays live progress, ETA, and metrics in terminal
"""

import json
import time
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import subprocess

class TrainingMonitor:
    """Monitor and display real-time training progress"""
    
    def __init__(self, results_dir: Path = Path("training_results"), refresh_interval: int = 5):
        self.results_dir = Path(results_dir)
        self.refresh_interval = refresh_interval
        self.last_epochs = {}
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def format_duration(self, seconds: float) -> str:
        """Format seconds to readable duration"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def load_status(self, phase: int, num_speakers: int) -> Optional[dict]:
        """Load training status from JSON"""
        status_file = self.results_dir / f"phase_{phase}_{num_speakers}spk" / f"training_status_phase_{phase}_{num_speakers}spk.json"
        
        if not status_file.exists():
            return None
        
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def display_phase_status(self, phase: int, num_speakers: int):
        """Display status for a specific phase"""
        status = self.load_status(phase, num_speakers)
        
        print(f"\n{'='*80}")
        print(f"PHASE {phase}: {num_speakers}-Speaker Separation")
        print(f"{'='*80}\n")
        
        if status is None:
            print("  ⏳ Waiting for phase to start...\n")
            return False
        
        # Header info
        print(f"  Status:        {status['status']}")
        print(f"  GPUs:          {status['gpus']}")
        print(f"  Start Time:    {status['start_time']}")
        
        if status['status'] in ['COMPLETED', 'INTERRUPTED']:
            print(f"  End Time:      {status['end_time']}")
            print(f"  Total Time:    {status.get('total_training_time_hours', 'N/A')} hours")
        
        # Training progress
        history = status.get('training_history', {})
        
        if len(history.get('epoch', [])) == 0:
            print("\n  ⏳ Training initializing...\n")
            return True
        
        current_epoch = status['current_epoch']
        total_epochs = status['total_epochs']
        best_val_sisnr = status['best_val_sisnr']
        best_epoch = status['best_val_sisnr_epoch']
        
        print(f"\n  Current Epoch:     {current_epoch + 1} / {total_epochs}")
        
        # Progress bar
        if total_epochs:
            progress = (current_epoch + 1) / total_epochs
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  Progress:        [{bar}] {progress*100:.1f}%")
        
        # Metrics from latest epoch
        if len(history['epoch']) > 0:
            last_idx = -1
            last_epoch = history['epoch'][last_idx]
            train_loss = history['train_loss'][last_idx]
            train_sisnr = history['train_sisnr'][last_idx]
            val_loss = history['val_loss'][last_idx] if len(history.get('val_loss', [])) > 0 else None
            val_sisnr = history['val_sisnr'][last_idx] if len(history.get('val_sisnr', [])) > 0 else None
            lr = history['learning_rate'][last_idx]
            grad_norm = history['grad_norm'][last_idx]
            epoch_time = history['epoch_time'][last_idx]
            
            print(f"\n  Latest Metrics:")
            print(f"    Train Loss:    {train_loss:.4f}")
            print(f"    Train SI-SNR:  {train_sisnr:.2f} dB")
            if val_loss is not None:
                print(f"    Val Loss:      {val_loss:.4f}")
            if val_sisnr is not None:
                print(f"    Val SI-SNR:    {val_sisnr:.2f} dB")
            print(f"    Learn Rate:    {lr:.2e}")
            print(f"    Grad Norm:     {grad_norm:.4f}")
            print(f"    Epoch Time:    {self.format_duration(epoch_time)}")
            
            print(f"\n  Best Performance:")
            print(f"    Best Val SI-SNR: {best_val_sisnr:.2f} dB (Epoch {best_epoch})")
        
        # ETA
        if status.get('eta_hours') is not None:
            eta_hours = status['eta_hours']
            print(f"\n  Estimated Time Remaining: {eta_hours:.1f} hours")
        
        print()
        return True
    
    def display_all_phases(self):
        """Display status for all phases"""
        self.clear_screen()
        
        print(f"\n╔{'='*78}╗")
        print(f"║  DPRNN-TasNet Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'':>37}║")
        print(f"║  GPU 5 & 6 - Multi-Speaker Separation{'':>36}║")
        print(f"╚{'='*78}╝")
        
        phases = [
            (1, 2, "2-speaker"),
            (2, 3, "3-speaker"),
            (3, 4, "4-speaker"),
            (4, 5, "5-speaker - FINAL TARGET")
        ]
        
        any_running = False
        for phase, num_speakers, label in phases:
            if self.display_phase_status(phase, num_speakers):
                any_running = True
        
        if not any_running:
            print("\n  No training processes detected.")
            print("  Start training with: python train_enhanced.py --full-curriculum\n")
        else:
            print(f"  Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Refresh rate: Every {self.refresh_interval} seconds")
            print("  Press Ctrl+C to exit\n")
    
    def watch(self):
        """Continuously monitor training"""
        try:
            while True:
                self.display_all_phases()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

def generate_summary_report(results_dir: Path = Path("training_results")) -> str:
    """Generate a summary report of all phases"""
    report = []
    report.append("\n" + "="*80)
    report.append("DPRNN TRAINING SUMMARY REPORT")
    report.append("="*80)
    
    phases = [(1, 2), (2, 3), (3, 4), (4, 5)]
    
    for phase, num_speakers in phases:
        status_file = results_dir / f"phase_{phase}_{num_speakers}spk" / f"training_status_phase_{phase}_{num_speakers}spk.json"
        
        if not status_file.exists():
            continue
        
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        report.append(f"\nPhase {phase}: {num_speakers}-Speaker Separation")
        report.append("-" * 40)
        report.append(f"  Status:              {status['status']}")
        report.append(f"  Best Val SI-SNR:     {status['best_val_sisnr']:.2f} dB")
        report.append(f"  Best Epoch:          {status['best_val_sisnr_epoch']}")
        if 'total_training_time_hours' in status:
            report.append(f"  Total Training Time: {status['total_training_time_hours']} hours")
    
    report.append("\n" + "="*80 + "\n")
    return "\n".join(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor DPRNN training progress")
    parser.add_argument("--watch", action="store_true", help="Continuously watch training")
    parser.add_argument("--summary", action="store_true", help="Show summary report")
    parser.add_argument("--results-dir", type=str, default="training_results", help="Results directory")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(Path(args.results_dir), args.refresh)
    
    if args.watch:
        monitor.watch()
    elif args.summary:
        print(generate_summary_report(Path(args.results_dir)))
    else:
        monitor.display_all_phases()
