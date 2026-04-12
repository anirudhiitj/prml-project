#!/usr/bin/env python3
"""
Speech Separation GUI — Drop an MP4/audio file → get 2 separated speaker WAVs.

Uses the C++ inference binary under the hood:
  1. ffmpeg converts input → 8kHz mono WAV
  2. ./build/inference separates into 2 speakers
  3. Results shown in GUI with playback buttons

Requirements: Python 3 + tkinter (built-in), ffmpeg, aplay/paplay (for playback)
"""

import os
import sys
import subprocess
import threading
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent
INFERENCE_BIN = PROJECT_DIR / "build" / "inference"
CHECKPOINT    = PROJECT_DIR / "checkpoints" / "best_tasnet.pt"
LIBTORCH_LIB  = PROJECT_DIR / "libtorch" / "lib"
OUTPUT_DIR    = PROJECT_DIR / "output"

SAMPLE_RATE   = 8000
SUPPORTED_EXT = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flac",
                 ".mp3", ".m4a", ".ogg", ".wav", ".aac", ".wma"}


def get_checkpoint():
    """Return the best available checkpoint path."""
    if CHECKPOINT.exists():
        return CHECKPOINT
    alt = PROJECT_DIR / "checkpoints" / "latest_tasnet.pt"
    if alt.exists():
        return alt
    return None


def check_dependencies():
    """Check that all required tools are available."""
    issues = []
    if not INFERENCE_BIN.exists():
        issues.append(f"Inference binary not found at:\n  {INFERENCE_BIN}\n  → Run: cd build && cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make -j$(nproc)")
    if get_checkpoint() is None:
        issues.append("No checkpoint found in checkpoints/\n  → Train the model first")
    if shutil.which("ffmpeg") is None:
        issues.append("ffmpeg not found.\n  → sudo apt install ffmpeg")
    return issues


def convert_to_wav(input_path: str, output_wav: str) -> tuple[bool, str]:
    """Convert any audio/video file to 8kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                    # discard video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(SAMPLE_RATE),  # 8kHz
        "-ac", "1",               # mono
        output_wav
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return False, r.stderr[-500:] if r.stderr else "ffmpeg failed"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out (>120s)"
    except Exception as e:
        return False, str(e)


def run_inference(wav_path: str, out_dir: str) -> tuple[bool, str]:
    """Run the C++ inference binary."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LIBTORCH_LIB) + ":" + env.get("LD_LIBRARY_PATH", "")

    ckpt = get_checkpoint()
    cmd = [
        str(INFERENCE_BIN),
        "--model", "tasnet",
        "--checkpoint", str(ckpt),
        "--input", wav_path,
        "--output_dir", out_dir,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if r.returncode != 0:
            err = r.stderr[-500:] if r.stderr else r.stdout[-500:]
            return False, err or "Inference failed"
        return True, r.stdout
    except subprocess.TimeoutExpired:
        return False, "Inference timed out (>300s)"
    except Exception as e:
        return False, str(e)


def play_audio(wav_path: str):
    """Play a WAV file using whatever player is available."""
    players = ["paplay", "aplay", "ffplay -nodisp -autoexit"]
    for p in players:
        parts = p.split()
        if shutil.which(parts[0]):
            subprocess.Popen(parts + [wav_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    messagebox.showwarning("Playback", f"No audio player found.\nFile saved at:\n{wav_path}")


# ── GUI ───────────────────────────────────────────────────────────────────────

class SeparationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speech Separation")
        self.root.geometry("620x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.current_file = None
        self.out_dir = str(OUTPUT_DIR)
        self.source_paths = []

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────

    def _build_ui(self):
        bg     = "#1e1e2e"
        fg     = "#cdd6f4"
        accent = "#89b4fa"
        green  = "#a6e3a1"
        card   = "#313244"
        btn_bg = "#45475a"

        # Title
        tk.Label(self.root, text="🎙 Speech Separation", font=("Helvetica", 20, "bold"),
                 bg=bg, fg=accent).pack(pady=(18, 2))
        tk.Label(self.root, text="Drop an MP4 / audio file → 2 separated speakers",
                 font=("Helvetica", 10), bg=bg, fg="#6c7086").pack()

        # ── Drop zone ────────────────────────────────────────────────────
        self.drop_frame = tk.Frame(self.root, bg=card, highlightbackground="#585b70",
                                   highlightthickness=2, cursor="hand2")
        self.drop_frame.pack(padx=30, pady=(18, 10), fill="x", ipady=28)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="📂  Click here to select a file\n\nSupports: MP4, MKV, AVI, MP3, WAV, FLAC …",
            font=("Helvetica", 11), bg=card, fg="#a6adc8", justify="center")
        self.drop_label.pack(expand=True)

        self.drop_frame.bind("<Button-1>", lambda e: self._pick_file())
        self.drop_label.bind("<Button-1>", lambda e: self._pick_file())

        # ── Separate button ──────────────────────────────────────────────
        self.sep_btn = tk.Button(
            self.root, text="✦  Separate Speakers", font=("Helvetica", 13, "bold"),
            bg=accent, fg="#1e1e2e", activebackground="#74c7ec",
            relief="flat", padx=20, pady=8, state="disabled",
            command=self._on_separate)
        self.sep_btn.pack(pady=(6, 10))

        # ── Status ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="")
        self.status_lbl = tk.Label(self.root, textvariable=self.status_var,
                                    font=("Helvetica", 10), bg=bg, fg="#f9e2af",
                                    wraplength=560, justify="center")
        self.status_lbl.pack()

        # ── Results area ─────────────────────────────────────────────────
        self.result_frame = tk.Frame(self.root, bg=bg)
        self.result_frame.pack(padx=30, pady=(10, 0), fill="x")

        # Speaker 1
        self.spk1_frame = tk.Frame(self.result_frame, bg=card, highlightbackground="#585b70",
                                    highlightthickness=1)
        self.spk1_frame.pack(fill="x", pady=4, ipady=6, ipadx=8)
        tk.Label(self.spk1_frame, text="🔊 Speaker 1", font=("Helvetica", 11, "bold"),
                 bg=card, fg=green).pack(side="left", padx=(12, 8))
        self.spk1_path = tk.Label(self.spk1_frame, text="—", font=("Helvetica", 9),
                                   bg=card, fg="#6c7086", anchor="w")
        self.spk1_path.pack(side="left", expand=True, fill="x")
        self.play1_btn = tk.Button(self.spk1_frame, text="▶ Play", font=("Helvetica", 9),
                                    bg=btn_bg, fg=fg, relief="flat", padx=8,
                                    state="disabled", command=lambda: self._play(0))
        self.play1_btn.pack(side="right", padx=(4, 6))
        self.save1_btn = tk.Button(self.spk1_frame, text="💾 Save As", font=("Helvetica", 9),
                                    bg=btn_bg, fg=fg, relief="flat", padx=8,
                                    state="disabled", command=lambda: self._save_as(0))
        self.save1_btn.pack(side="right", padx=(4, 0))

        # Speaker 2
        self.spk2_frame = tk.Frame(self.result_frame, bg=card, highlightbackground="#585b70",
                                    highlightthickness=1)
        self.spk2_frame.pack(fill="x", pady=4, ipady=6, ipadx=8)
        tk.Label(self.spk2_frame, text="🔊 Speaker 2", font=("Helvetica", 11, "bold"),
                 bg=card, fg=green).pack(side="left", padx=(12, 8))
        self.spk2_path = tk.Label(self.spk2_frame, text="—", font=("Helvetica", 9),
                                   bg=card, fg="#6c7086", anchor="w")
        self.spk2_path.pack(side="left", expand=True, fill="x")
        self.play2_btn = tk.Button(self.spk2_frame, text="▶ Play", font=("Helvetica", 9),
                                    bg=btn_bg, fg=fg, relief="flat", padx=8,
                                    state="disabled", command=lambda: self._play(1))
        self.play2_btn.pack(side="right", padx=(4, 6))
        self.save2_btn = tk.Button(self.spk2_frame, text="💾 Save As", font=("Helvetica", 9),
                                    bg=btn_bg, fg=fg, relief="flat", padx=8,
                                    state="disabled", command=lambda: self._save_as(1))
        self.save2_btn.pack(side="right", padx=(4, 0))

        # Info footer
        ckpt_name = get_checkpoint().name if get_checkpoint() else "none"
        tk.Label(self.root,
                 text=f"Model: Conv-TasNet (8.2M params) • 2 speakers • 8kHz • Checkpoint: {ckpt_name}",
                 font=("Helvetica", 8), bg=bg, fg="#585b70").pack(side="bottom", pady=(0, 8))

    # ── Actions ───────────────────────────────────────────────────────────

    def _pick_file(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        path = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=[("Audio/Video", exts), ("All files", "*.*")])
        if not path:
            return
        self.current_file = path
        name = os.path.basename(path)
        self.drop_label.config(text=f"📄  {name}", fg="#cdd6f4")
        self.sep_btn.config(state="normal")
        self.status_var.set("")
        self._reset_results()

    def _reset_results(self):
        self.source_paths = []
        self.spk1_path.config(text="—")
        self.spk2_path.config(text="—")
        for btn in (self.play1_btn, self.play2_btn, self.save1_btn, self.save2_btn):
            btn.config(state="disabled")

    def _set_status(self, msg, color="#f9e2af"):
        self.status_var.set(msg)
        self.status_lbl.config(fg=color)

    def _on_separate(self):
        if not self.current_file:
            return
        self.sep_btn.config(state="disabled")
        self._reset_results()
        self._set_status("⏳ Converting audio with ffmpeg …")
        threading.Thread(target=self._separate_thread, daemon=True).start()

    def _separate_thread(self):
        try:
            # Create output directory
            os.makedirs(self.out_dir, exist_ok=True)

            # Step 1: Convert to WAV
            tmp_wav = os.path.join(self.out_dir, "_input_converted.wav")
            ok, err = convert_to_wav(self.current_file, tmp_wav)
            if not ok:
                self.root.after(0, lambda: self._set_status(f"❌ ffmpeg error: {err}", "#f38ba8"))
                self.root.after(0, lambda: self.sep_btn.config(state="normal"))
                return

            self.root.after(0, lambda: self._set_status("⏳ Running Conv-TasNet inference …"))

            # Step 2: Inference
            ok, out = run_inference(tmp_wav, self.out_dir)
            if not ok:
                self.root.after(0, lambda: self._set_status(f"❌ Inference error: {out}", "#f38ba8"))
                self.root.after(0, lambda: self.sep_btn.config(state="normal"))
                return

            # Step 3: Collect results
            s1 = os.path.join(self.out_dir, "source_1.wav")
            s2 = os.path.join(self.out_dir, "source_2.wav")
            if os.path.exists(s1) and os.path.exists(s2):
                self.source_paths = [s1, s2]
                self.root.after(0, self._show_results)
            else:
                self.root.after(0, lambda: self._set_status("❌ Output files not found", "#f38ba8"))

        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"❌ {e}", "#f38ba8"))
        finally:
            self.root.after(0, lambda: self.sep_btn.config(state="normal"))

    def _show_results(self):
        self._set_status("✅ Separation complete!", "#a6e3a1")
        self.spk1_path.config(text=self.source_paths[0], fg="#cdd6f4")
        self.spk2_path.config(text=self.source_paths[1], fg="#cdd6f4")
        for btn in (self.play1_btn, self.play2_btn, self.save1_btn, self.save2_btn):
            btn.config(state="normal")

    def _play(self, idx):
        if idx < len(self.source_paths):
            play_audio(self.source_paths[idx])

    def _save_as(self, idx):
        if idx >= len(self.source_paths):
            return
        dest = filedialog.asksaveasfilename(
            title=f"Save Speaker {idx+1}",
            defaultextension=".wav",
            initialfile=f"speaker_{idx+1}.wav",
            filetypes=[("WAV audio", "*.wav")])
        if dest:
            shutil.copy2(self.source_paths[idx], dest)
            self._set_status(f"💾 Saved to {dest}", "#a6e3a1")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    issues = check_dependencies()
    if issues:
        # Show issues in GUI instead of crashing
        root = tk.Tk()
        root.title("Speech Separation — Setup Error")
        root.geometry("500x300")
        root.configure(bg="#1e1e2e")
        tk.Label(root, text="⚠️  Setup Issues", font=("Helvetica", 16, "bold"),
                 bg="#1e1e2e", fg="#f38ba8").pack(pady=(20, 10))
        for issue in issues:
            tk.Label(root, text=f"• {issue}", font=("Helvetica", 10),
                     bg="#1e1e2e", fg="#cdd6f4", justify="left",
                     wraplength=450, anchor="w").pack(padx=30, pady=4, anchor="w")
        tk.Button(root, text="Quit", command=root.destroy,
                  bg="#45475a", fg="#cdd6f4", relief="flat", padx=20).pack(pady=20)
        root.mainloop()
        return

    root = tk.Tk()
    app = SeparationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
