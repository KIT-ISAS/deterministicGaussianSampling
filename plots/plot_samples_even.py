import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# plotting constants
GRID_RES = 400
SEED = "42"
BMAX = 5.0

# Figure A: standard normal approximated with more and more samples.
STANDARD_NORMAL_CASES = [
    {"sigma": 1.0, "L": 100},
    {"sigma": 1.0, "L": 200},
    {"sigma": 1.0, "L": 400},
]

# Figure B: isotropic sigma^2 * I at a fixed L
ISOTROPIC_CASES = [
    {"sigma": 0.5, "L": 100},
    {"sigma": 1.0, "L": 100},
    {"sigma": 2.0, "L": 100},
]

# Figure C: the two implementations side by side on the same problem.
COMPARISON_L = 400


def resolve_generator(path):
    for candidate in (path, path + ".exe"):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    sys.exit(f"error: generator not found at '{path}' (or '{path}.exe'). ")


def run_even_generator(generator, sigma, n_samples, n_dim=2):
    env = dict(os.environ, GSL_RNG_SEED=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "samples.csv")
        cmd = [
            generator,
            str(n_dim),
            str(n_samples),
            out_csv,
            "%.17g" % sigma,
            "%.17g" % BMAX,
        ]
        subprocess.run(cmd, env=env, check=True)
        samples = np.loadtxt(out_csv, delimiter=",")
    return samples.reshape(n_samples, n_dim)


def run_short_generator(generator, sigma, n_samples, n_dim=2):
    """Invoke the existing quadrature sampler, for the comparison figure."""
    env = dict(os.environ, GSL_RNG_SEED=SEED)
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = os.path.join(tmp, "samples.csv")
        cmd = [generator, str(n_dim), str(n_samples), out_csv]
        cmd += ["%.17g" % sigma] * n_dim
        cmd += [str(int(BMAX))]
        subprocess.run(cmd, env=env, check=True)
        samples = np.loadtxt(out_csv, delimiter=",")
    return samples.reshape(n_samples, n_dim)


def gaussian_density(sigma, axis_limit):
    axis = np.linspace(-axis_limit, axis_limit, GRID_RES)
    xx, yy = np.meshgrid(axis, axis)
    quad = (xx * xx + yy * yy) / (sigma * sigma)
    norm = 1.0 / (2.0 * np.pi * sigma * sigma)
    return norm * np.exp(-0.5 * quad)


def draw_panel(ax, points, sigma, title, axis_limit=None):
    if axis_limit is None:
        axis_limit = 4.0 * sigma

    ax.imshow(
        gaussian_density(sigma, axis_limit),
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
    ax.set_title(title, fontsize=11)


def build_standard_normal_figure(generator, out_path):
    fig, axes = plt.subplots(
        1, len(STANDARD_NORMAL_CASES), figsize=(4.2 * len(STANDARD_NORMAL_CASES), 4.8)
    )
    fig.patch.set_facecolor("white")
    for ax, case in zip(np.atleast_1d(axes), STANDARD_NORMAL_CASES):
        points = run_even_generator(generator, case["sigma"], case["L"])
        draw_panel(ax, points, case["sigma"], "L = %d\nΣ = I" % case["L"])
    fig.suptitle(
        "Closed-form LCD (N = 2): standard normal with increasing sample counts",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def build_isotropic_figure(generator, out_path):
    fig, axes = plt.subplots(
        1, len(ISOTROPIC_CASES), figsize=(4.2 * len(ISOTROPIC_CASES), 4.8)
    )
    fig.patch.set_facecolor("white")

    axis_limit = 3.0 * max(case["sigma"] for case in ISOTROPIC_CASES)

    for ax, case in zip(np.atleast_1d(axes), ISOTROPIC_CASES):
        sigma = case["sigma"]
        points = run_even_generator(generator, sigma, case["L"])
        radius = np.hypot(points[:, 0], points[:, 1]).max()
        draw_panel(
            ax,
            points,
            sigma,
            "L = %d,  σ = %g\nΣ = %g · I" % (case["L"], sigma, sigma * sigma),
            axis_limit=axis_limit,
        )
    fig.suptitle(
        "Closed-form LCD (N = 2): isotropic Σ = σ² I",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())

    out_path = out_path.replace(".png", "_scaled_axes.png")
    fig, axes = plt.subplots(
            1, len(ISOTROPIC_CASES), figsize=(4.2 * len(ISOTROPIC_CASES), 4.8)
        )
    fig.patch.set_facecolor("white")

    for ax, case in zip(np.atleast_1d(axes), ISOTROPIC_CASES):
        sigma = case["sigma"]
        points = run_even_generator(generator, sigma, case["L"])
        # radius = np.hypot(points[:, 0], points[:, 1]).max()
        draw_panel(
            ax,
            points,
            sigma,
            "L = %d,  σ = %g\nΣ = %g · I" % (case["L"], sigma, sigma * sigma),
        )
    fig.suptitle(
        "Closed-form LCD (N = 2): isotropic Σ = σ² I",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    
    plt.close(fig)
    print(f"wrote {out_path}")


def build_comparison_figure(even_generator, short_generator, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8))
    fig.patch.set_facecolor("white")

    closed_form = run_even_generator(even_generator, 1.0, COMPARISON_L)
    quadrature = run_short_generator(short_generator, 1.0, COMPARISON_L)

    for ax, points, label in (
        (axes[0], closed_form, "closed form (new)"),
        (axes[1], quadrature, "quadrature (existing)"),
    ):
        covariance = np.cov(points.T, bias=True)
        draw_panel(
            ax,
            points,
            1.0,
            "%s\ntr(Cov)/2 = %.4f" % (label, np.trace(covariance) / 2.0),
        )

    fig.suptitle(
        "Closed-form vs quadrature LCD (N = 2, L = %d, bMax = %g, same seed)"
        % (COMPARISON_L, BMAX),
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        default=os.path.join("build", "plots", "generate_samples_even"),
        help="path to the compiled generate_samples_even binary",
    )
    parser.add_argument(
        "--short-generator",
        default=os.path.join("build", "plots", "generate_samples"),
        help="path to the compiled generate_samples binary (comparison figure)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("doxygen", "images"),
        help="directory to write the PNG figures into",
    )
    args = parser.parse_args()

    generator = resolve_generator(args.generator)
    short_generator = resolve_generator(args.short_generator)
    os.makedirs(args.out_dir, exist_ok=True)

    build_standard_normal_figure(
        generator, os.path.join(args.out_dir, "samples_even_standard_normal.png")
    )
    build_isotropic_figure(
        generator, os.path.join(args.out_dir, "samples_even_isotropic.png")
    )
    build_comparison_figure(
        generator,
        short_generator,
        os.path.join(args.out_dir, "samples_even_vs_quadrature.png"),
    )


if __name__ == "__main__":
    main()
