#!/usr/bin/env python3
"""
Plot the sparsity and magnitude structure of Kore generalized eigenproblem matrices.

Typical use after an assemble step:

    python3 tools/plot_matrix_structure.py --max-n-display 4000

From a notebook:

    from tools.plot_matrix_structure import plot_eigen_matrix_structure
    plot_eigen_matrix_structure(kore=True, max_n_display=4000)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kore")

import matplotlib
if "--save" in sys.argv or any(arg.startswith("--save=") for arg in sys.argv):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss


def load_kore_csr_npz(path):
    """Load Kore's CSR `.npz` matrices, with scipy `.npz` fallback."""
    path = Path(path)
    try:
        return ss.load_npz(path).tocsr()
    except Exception:
        data = np.load(path)
        return ss.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"]),
        )


def _matrix_from_input(source, name):
    if source is None:
        return load_kore_csr_npz(f"{name}.npz")

    if ss.issparse(source):
        return source.tocsr()

    if isinstance(source, (str, Path)):
        return load_kore_csr_npz(source)

    return ss.csr_matrix(np.asarray(source))


def _matrices_from_result(result=None, B=None, kore=False):
    if kore:
        return load_kore_csr_npz("A.npz"), load_kore_csr_npz("B.npz"), None

    if isinstance(result, dict):
        params = result.get("params", {})
        return (
            _matrix_from_input(result["A"], "A"),
            _matrix_from_input(result["B"], "B"),
            params,
        )

    if isinstance(result, (tuple, list)) and len(result) == 2 and B is None:
        return _matrix_from_input(result[0], "A"), _matrix_from_input(result[1], "B"), None

    return _matrix_from_input(result, "A"), _matrix_from_input(B, "B"), None


def _threshold_sparse(matrix, tol):
    matrix = matrix.tocoo(copy=True)
    if tol > 0 and matrix.nnz:
        keep = np.abs(matrix.data) > tol
        matrix = ss.coo_matrix(
            (matrix.data[keep], (matrix.row[keep], matrix.col[keep])),
            shape=matrix.shape,
        )
    return matrix.tocsr()


def _crop_matrix(matrix, max_n_display):
    if max_n_display is None:
        return matrix
    n = min(max_n_display, matrix.shape[0], matrix.shape[1])
    return matrix[:n, :n]


def _kore_block_boundaries(matrix_shape):
    root = Path.cwd()
    bin_path = root / "bin"
    if str(bin_path) not in sys.path:
        sys.path.insert(0, str(bin_path))

    try:
        import utils as ut
    except Exception:
        return []

    raw_blocks = [
        ("u", ut.offset_u, ut.size_u),
        ("v", ut.offset_v, ut.size_v),
        ("f", ut.offset_f, ut.size_f),
        ("g", ut.offset_g, ut.size_g),
        ("rot", ut.offset_rotdyn, ut.size_rotdyn),
        ("fic", ut.offset_fic, ut.size_fic),
        ("gic", ut.offset_gic, ut.size_gic),
        ("p", ut.offset_p, ut.size_p),
        ("q", ut.offset_q, ut.size_q),
        ("theta", ut.offset_h, ut.size_h),
        ("xi", ut.offset_i, ut.size_i),
    ]

    nrows, ncols = matrix_shape
    blocks = []
    for label, start, size in raw_blocks:
        stop = start + size
        if size > 0 and start < nrows and start < ncols:
            blocks.append((label, start, min(stop, nrows, ncols)))
    return blocks


def _result_block_boundaries(params, matrix_shape):
    if not params or "n" not in params:
        return []
    n = int(params["n"])
    nmax = min(matrix_shape)
    return [("F", 0, min(n, nmax)), ("G", n, min(2*n, nmax))]


def _draw_block_guides(ax, blocks, n_display, block_labels="top"):
    top_labels = []
    for label, start, stop in blocks:
        if start >= n_display:
            continue
        clipped_stop = min(stop, n_display)
        if start > 0:
            ax.axhline(start - 0.5, color="tab:red", linestyle="--", linewidth=0.5, alpha=0.45)
            ax.axvline(start - 0.5, color="tab:red", linestyle="--", linewidth=0.5, alpha=0.45)
        center = 0.5*(start + clipped_stop)
        if block_labels != "none" and clipped_stop - start > 20:
            top_labels.append((center, label))
            if block_labels == "both":
                ax.text(-0.018*n_display, center, label, ha="right", va="center", fontsize=7, alpha=0.8)

    if block_labels != "none" and top_labels:
        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xticks([center for center, _ in top_labels])
        top_ax.set_xticklabels([label for _, label in top_labels])
        top_ax.tick_params(axis="x", labelsize=7, length=0, pad=8)
        top_ax.spines["top"].set_visible(False)


def _plot_sparse_magnitude(ax, matrix, max_points):
    coo = matrix.tocoo()
    if coo.nnz == 0:
        ax.text(0.5, 0.5, "no entries above tol", ha="center", va="center", transform=ax.transAxes)
        return None

    if max_points is not None and coo.nnz > max_points:
        idx = np.linspace(0, coo.nnz - 1, max_points, dtype=int)
        row = coo.row[idx]
        col = coo.col[idx]
        data = coo.data[idx]
    else:
        row = coo.row
        col = coo.col
        data = coo.data

    color = np.log10(np.abs(data))
    size = max(0.2, min(4.0, 25000.0 / max(1, matrix.shape[0])))
    return ax.scatter(col, row, c=color, s=size, cmap="viridis", marker="s", linewidths=0)


def plot_eigen_matrix_structure(
    result=None,
    B=None,
    tol=1e-12,
    max_n_display=None,
    kore=False,
    show_blocks=True,
    block_labels="top",
    max_points=500_000,
    save=None,
):
    """
    Plot the sparsity structure of a generalized eigenproblem.

    This function accepts either:

    - the small notebook dictionary returned by `solve_radial_FG_eigenproblem`;
    - two matrices, `plot_eigen_matrix_structure(A, B)`;
    - two paths, `plot_eigen_matrix_structure("A.npz", "B.npz")`;
    - the full Kore assembled matrices with `plot_eigen_matrix_structure(kore=True)`.

    For full Kore runs, `max_n_display` is useful because the complete matrices can
    contain hundreds of thousands of rows. If it is `None`, the full sparse pattern
    is plotted, but this can be slow for high-resolution runs.
    """
    A, Bmat, params = _matrices_from_result(result, B, kore=kore)
    if A.shape != Bmat.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} and {Bmat.shape}")

    A_plot = _threshold_sparse(_crop_matrix(A, max_n_display), tol)
    B_plot = _threshold_sparse(_crop_matrix(Bmat, max_n_display), tol)
    n_display = A_plot.shape[0]

    if show_blocks:
        if params is not None:
            blocks = _result_block_boundaries(params, A.shape)
        else:
            blocks = _kore_block_boundaries(A.shape)
    else:
        blocks = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    markersize = max(0.2, min(2.0, 12000.0 / max(1, n_display)))
    axes[0].spy(A_plot, markersize=markersize, color="black")
    axes[0].set_title("A pattern", fontsize=10, pad=18)

    axes[1].spy(B_plot, markersize=markersize, color="black")
    axes[1].set_title("B pattern", fontsize=10, pad=18)

    image = _plot_sparse_magnitude(axes[2], A_plot, max_points=max_points)
    axes[2].set_title(r"$|A_{ij}|$", fontsize=10, pad=18)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.035)
        colorbar.ax.tick_params(labelsize=8)

    for i, ax in enumerate(axes):
        _draw_block_guides(ax, blocks, n_display, block_labels=block_labels)
        ax.set_xlim(-0.5, A_plot.shape[1] - 0.5)
        ax.set_ylim(A_plot.shape[0] - 0.5, -0.5)
        ax.set_xlabel("column", fontsize=9)
        if i == 0:
            ax.set_ylabel("row", fontsize=9)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
        )

    title = "Kore matrix structure" if params is None else "Ultraspherical radial matrix structure"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.8, rect=(0, 0, 1, 0.94))

    if save is not None:
        fig.savefig(save, dpi=220)
        print(f"wrote {save}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("A", nargs="?", default="A.npz")
    parser.add_argument("B", nargs="?", default="B.npz")
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--max-n-display", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--no-blocks", action="store_true")
    parser.add_argument("--block-labels", choices=["none", "top", "both"], default="top")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    plot_eigen_matrix_structure(
        args.A,
        args.B,
        tol=args.tol,
        max_n_display=args.max_n_display,
        show_blocks=not args.no_blocks,
        block_labels=args.block_labels,
        max_points=args.max_points,
        save=args.save,
    )
    if args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
