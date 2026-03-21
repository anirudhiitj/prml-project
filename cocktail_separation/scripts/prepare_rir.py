from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a bank of synthetic RIRs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(args.count), desc="rirs"):
        dims = rng.uniform(3.0, 10.0, size=3)
        dims[2] = rng.uniform(2.5, 4.5)
        absorption = float(rng.uniform(0.1, 0.5))

        room = pra.ShoeBox(
            p=dims,
            fs=args.sample_rate,
            materials=pra.Material(absorption),
            max_order=12,
        )

        mic = rng.uniform(low=[0.5, 0.5, 1.2], high=[dims[0] - 0.5, dims[1] - 0.5, 2.0])
        src = rng.uniform(low=[0.5, 0.5, 1.2], high=[dims[0] - 0.5, dims[1] - 0.5, 2.0])

        room.add_microphone_array(np.array(mic).reshape(3, 1))
        room.add_source(src)
        room.compute_rir()

        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        sf.write(out_dir / f"rir_{idx:05d}.wav", rir, args.sample_rate)


if __name__ == "__main__":
    main()
