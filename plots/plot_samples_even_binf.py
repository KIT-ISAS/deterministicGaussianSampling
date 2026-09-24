import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_RES = 400
SEED = "42"

STANDARD_NORMAL_CASES = [
    {"L": 100},
    {"L": 200},
    {"L": 400},
]

DISTANCE_RE = re.compile(r"distance\s*=\s*([-+0-9.eE]+)")


def resolve_generator(path):
    for candidate in (path, path + ".exe"):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    sys.exit(f"error: generator not found at '{path}' (or '{path}.exe'). ")


def run_binf_generator(generator, n_samples, n_dim=2):
    """Invoke the bMax-free sampler."""
    env = dict(os.environ, GSL_RNG_SEED=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "samples.csv")
        cmd = [generator, str(n_dim), str(n_samples), out_csv]
        done = subprocess.run(
            cmd, env=env, check=True, stderr=subprocess.PIPE, text=True
        )
        samples = np.loadtxt(out_csv, delimiter=",")

    match = DISTANCE_RE.search(done.stderr or "")
    distance = float(match.group(1)) if match else float("nan")
    return samples.reshape(n_samples, n_dim), distance


def gaussian_density(axis_limit):
    axis = np.linspace(-axis_limit, axis_limit, GRID_RES)
    xx, yy = np.meshgrid(axis, axis)
    return np.exp(-0.5 * (xx * xx + yy * yy)) / (2.0 * np.pi)


def draw_panel(ax, points, title, axis_limit=4.0):
    ax.imshow(
        gaussian_density(axis_limit),
        origin="lower",
        extent=[-axis_limit, axis_limit, -axis_limit, axis_limit],
        vmin=0.0,
        aspect="equal",
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        c="#e8000b",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ticks = np.linspace(-axis_limit, axis_limit, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(title, fontsize=10.5)


def build_standard_normal_figure(generator, out_path):
    fig, axes = plt.subplots(
        1, len(STANDARD_NORMAL_CASES), figsize=(4.2 * len(STANDARD_NORMAL_CASES), 5.1)
    )
    fig.patch.set_facecolor("white")

    mean_norms = []
    for ax, case in zip(np.atleast_1d(axes), STANDARD_NORMAL_CASES):
        points, distance = run_binf_generator(generator, case["L"])
        covariance = np.cov(points.T, bias=True)
        mean_norms.append(np.linalg.norm(points.mean(axis=0)))
        draw_panel(
            ax,
            points,
            "L = %d\nΣ = I\nD = %.3e,  tr(Cov)/2 = %.4f"
            % (case["L"], distance, np.trace(covariance) / 2.0),
        )

    fig.suptitle(
        "bMax-free closed-form LCD (N = 2): standard normal with increasing "
        "sample count",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        default=os.path.join("build", "plots", "generate_samples_even_binf"),
        help="path to the compiled generate_samples_even_binf binary",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("doxygen", "images"),
        help="directory to write the PNG figure into",
    )
    args = parser.parse_args()

    generator = resolve_generator(args.generator)
    os.makedirs(args.out_dir, exist_ok=True)

    build_standard_normal_figure(
        generator,
        os.path.join(args.out_dir, "samples_even_binf_standard_normal.png"),
    )


if __name__ == "__main__":
    main()
