import argparse
import os
import torch
import torchaudio
import soundfile as sf
from model import SpeechSeparator

def separate_audio(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SpeechSeparator(n_src=args.n_src, sample_rate=args.sample_rate,
                            n_blocks=args.n_blocks, n_repeats=args.n_repeats,
                            bn_chan=args.bn_chan, hid_chan=args.hid_chan)
    
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"Loading weights from {args.ckpt_path}")
        # Assuming torch.save(model.state_dict(), path) was used
        state_dict = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: No checkpoint provided or found, using initialized weights!")
        
    model = model.to(device)
    model.eval()
    
    # Load input audio mixed file
    print(f"Processing {args.input_path}...")
    if not os.path.exists(args.input_path):
        print(f"Error: {args.input_path} does not exist.")
        return
        
    wav, sr = torchaudio.load(args.input_path)
    
    # Resample if needed
    if sr != args.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
        
    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # Run inference
    with torch.no_grad():
        wav = wav.to(device)
        # model expects (batch, channels, time) or (batch, time)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0) # Shape: (batch=1, channels=1, time)
            
        estimates = model(wav)  # Shape: (batch, n_src, time)
        
    estimates = estimates.squeeze(0).cpu() # Shape: (n_src, time)
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]
    
    for i in range(args.n_src):
        out_path = os.path.join(args.out_dir, f"{base_name}_speaker_{i+1}.wav")
        # Ensure shape is (channels, time) for torchaudio
        out_wav = estimates[i].unsqueeze(0)
        torchaudio.save(out_path, out_wav, args.sample_rate)
        print(f"Saved speaker {i+1} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conv-TasNet Speech Separation Inference")
    parser.add_argument("--input_path", type=str, required=True, help="Path to mixed audio file")
    parser.add_argument("--out_dir", type=str, default="separated_outputs", help="Directory to save separated files")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--n_src", type=int, default=2, help="Number of speakers to separate")
    parser.add_argument("--sample_rate", type=int, default=8000, help="Sample rate")
    
    # Dynamic architecture parameters
    parser.add_argument("--n_blocks", type=int, default=8, help="Conv-TasNet X (Blocks)")
    parser.add_argument("--n_repeats", type=int, default=3, help="Conv-TasNet R (Repeats)")
    parser.add_argument("--bn_chan", type=int, default=128, help="Conv-TasNet B (Bottleneck channels)")
    parser.add_argument("--hid_chan", type=int, default=512, help="Conv-TasNet H (Hidden channels)")
    args = parser.parse_args()
    
    separate_audio(args)
