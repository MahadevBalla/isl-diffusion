"""
Utility for plotting training loss curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EXPERIMENTS


def plot_loss_curves(
    exp_names: list[str], out_path: str = "./experiments/loss_curves.png"
) -> None:
    """Plots training loss curves for one or more experiments."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in exp_names:
        cfg = EXPERIMENTS[name]
        loss_path = Path(cfg.output_dir) / "loss_history.npy"
        if not loss_path.exists():
            print(f"[plot-curves] skipping '{name}': no loss_history.npy found")
            continue
        history = np.load(loss_path, allow_pickle=True)
        steps = [entry["step"] for entry in history]
        losses = [entry["loss"] for entry in history]

        ax.plot(steps, losses, label=name, alpha=0.85)

        fig_i, ax_i = plt.subplots(figsize=(9, 5))
        ax_i.plot(steps, losses, color="tab:blue")
        ax_i.set_xlabel("Training step")
        ax_i.set_ylabel("MSE loss")
        ax_i.set_title(f"{name}: training loss vs. step")
        ax_i.grid(True, alpha=0.3)
        plt.tight_layout()
        individual_path = Path(cfg.output_dir) / "loss_curve.png"
        fig_i.savefig(individual_path, dpi=150)
        plt.close(fig_i)
        print(f"[plot-curves] saved {name}'s individual curve to {individual_path}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss vs. step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot-curves] saved comparison to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curves.")
    parser.add_argument("--exps", nargs="+", required=True, choices=EXPERIMENTS.keys())
    parser.add_argument("--out", default="./experiments/loss_curves.png")
    args = parser.parse_args()
    plot_loss_curves(args.exps, out_path=args.out)
