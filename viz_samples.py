#!/usr/bin/env python3
"""Visualize PGM sample grids.

Usage:
    python3 viz_samples.py                         # show samples/heun_samples.pgm
    python3 viz_samples.py samples/ddim_samples.pgm
    python3 viz_samples.py samples/heun_samples.pgm out.png   # save as PNG
"""
import sys
import numpy as np


def load_pgm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5", f"expected P5, got {magic}"
        line = f.readline().strip()
        while line.startswith(b"#"):
            line = f.readline().strip()
        W, H = map(int, line.split())
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(H, W)
    return data


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/heun_samples.pgm"
    img = load_pgm(path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available — image loaded ({img.shape}), "
              "install matplotlib to display/save")
        sys.exit(0)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title(path)
    if len(sys.argv) > 2:
        plt.savefig(sys.argv[2], bbox_inches="tight", dpi=150)
        print(f"saved {sys.argv[2]}")
    else:
        plt.show()
