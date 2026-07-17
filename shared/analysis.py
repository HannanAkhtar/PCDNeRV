"""
Analysis utilities for PCD-NeRV v2: result loading, plots, and text summary.

Method-aware: baseline / prox_gl / weighted_sum / pcd runs are colored and
labeled separately so cross-method comparisons in one output dir are readable.
Group sparsity shown here is the channel-group diagnostic; physical-pruning
reports are the source of truth for compression numbers.
"""

import os
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("baseline", "prox_gl", "weighted_sum", "pcd")

C = {
    "baseline": "#6B7280",
    "prox_gl": "#059669",
    "weighted_sum": "#D97706",
    "pcd": "#2563EB",
}


# ── Result loading ────────────────────────────────────────────────────────────

def load_results(results_dir):
    """{filename_stem -> training-log dict} for all completed runs."""
    R = {}
    if not os.path.isdir(results_dir):
        return R
    for fn in sorted(os.listdir(results_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(results_dir, fn)) as f:
                obj = json.load(f)
            # training logs only — results/ also holds *_physical_pruning.json
            # side-products, which carry the run config but no epoch history
            if (obj.get("config", {}).get("method") in METHODS
                    and "final" in obj and "epochs" in obj):
                R[fn[:-5]] = obj
    return R


def _ex(log, k):
    return [r.get(k, float("nan")) for r in log["epochs"]]


def _method_color(log):
    return C.get(log.get("config", {}).get("method", "pcd"), C["pcd"])


def _run_label(name, log):
    cfg = log.get("config", {})
    m = cfg.get("method", "?")
    bits = [m]
    if cfg.get("tau") is not None:
        bits.append(f"τ={cfg['tau']}")
    if cfg.get("lambda_prox") is not None:
        bits.append(f"λp={cfg['lambda_prox']:.0e}")
    if cfg.get("ws_weight") is not None:
        bits.append(f"w={cfg['ws_weight']:.0e}")
    bits.append(f"s{cfg.get('seed', '?')}")
    return " ".join(bits)


def _vid_title(log):
    cfg = log.get("config", {})
    return f"{cfg.get('arch', 'nerv').upper()} / {cfg.get('vid', 'unknown')}"


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pareto(R, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    titles = {_vid_title(log) for log in R.values()}
    ax.set_title(
        "Pareto (diagnostic): PSNR vs Channel-Group Sparsity\n"
        + ", ".join(sorted(titles)),
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Channel-group sparsity (%)", fontsize=12)
    ax.set_ylabel("PSNR seen (dB)", fontsize=12)

    by_method = {}
    for name, log in R.items():
        f = log["final"]
        m = log["config"].get("method", "pcd")
        by_method.setdefault(m, []).append(
            (f.get("group_sparsity", 0), f.get("pred_seen_psnr", 0), _run_label(name, log)))

    for m, pts in sorted(by_method.items()):
        color = C.get(m, "#2563EB")
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=m,
                   marker="o", s=110, color=color, edgecolors="white",
                   linewidth=0.8, zorder=5)
        for sp, psnr, lbl in pts:
            ax.annotate(lbl, (sp, psnr), fontsize=7, textcoords="offset points",
                        xytext=(6, -10), color=color, alpha=0.85)
        if len(pts) > 1:
            pts_s = sorted(pts, key=lambda p: p[0])
            ax.plot([p[0] for p in pts_s], [p[1] for p in pts_s],
                    ls="-", alpha=0.35, color=color, zorder=1)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "plots", "pareto_front.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def plot_training(R, out):
    keys = sorted(R.keys())
    if not keys:
        return
    key = keys[len(keys) // 2]
    log = R[key]
    eps = _ex(log, "epoch")
    wu = log["config"].get("warmup", 0)
    color = _method_color(log)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f"Training Dynamics — {_vid_title(log)} — {key}",
                 fontsize=13, fontweight="bold")

    def shade(ax):
        if wu > 0:
            ax.axvspan(0, wu, alpha=0.08, color="orange")
            ax.axvline(wu, color="orange", ls=":", alpha=0.5)

    ax = axes[0, 0]
    shade(ax)
    ax.plot(eps, _ex(log, "primary_loss"), color=color, label="reconstruction loss")
    ax.set_title("Reconstruction Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    shade(ax)
    ax.plot(eps, _ex(log, "pred_seen_psnr"), color=color, label="seen")
    unseen = _ex(log, "pred_unseen_psnr")
    if any(not math.isnan(v) and v > 0 for v in unseen):
        ax.plot(eps, unseen, color=color, ls="--", alpha=0.7, label="unseen")
    ax.set_title("PSNR")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    shade(ax)
    ax.plot(eps, _ex(log, "group_sparsity"), color=color, label="Channel Groups")
    ax.plot(eps, _ex(log, "decoder_head_param_sparsity"), color="#059669", label="Decoder+Head Params")
    ax.plot(eps, _ex(log, "model_param_sparsity"), color="#9333EA", label="All Trainable Params")
    ax.set_title("Sparsity (diagnostic)")
    ax.set_ylabel("Sparsity (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    mu_vals = _ex(log, "mu")
    valid = [(e, v) for e, v in zip(eps, mu_vals) if not math.isnan(v)]
    ax = axes[1, 0]
    shade(ax)
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], color=color)
    ax.set_title("μ (GL correction strength; pcd only)")
    ax.grid(True, alpha=0.3)

    eff = _ex(log, "primary_efficiency")
    valid = [(e, v) for e, v in zip(eps, eff) if not math.isnan(v)]
    ax = axes[1, 1]
    shade(ax)
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], color=color)
    ax.axhline(1.0, color="grey", ls="--", alpha=0.5)
    ax.set_title("Primary Efficiency (pcd only)")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    cos_vals = _ex(log, "cosine_sim")
    valid = [(e, v) for e, v in zip(eps, cos_vals) if not math.isnan(v)]
    ax = axes[1, 2]
    shade(ax)
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], color="#8B5CF6")
    ax.set_title("cos(g̃_primary, g̃_GL) (pcd only)")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    p = os.path.join(out, "plots", "training_dynamics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def plot_psnr_all(R, out):
    fig, ax = plt.subplots(figsize=(12, 6))
    titles = {_vid_title(log) for log in R.values()}
    ax.set_title(f"PSNR (seen) — all runs\n{', '.join(sorted(titles))}",
                 fontsize=13, fontweight="bold")
    for name in sorted(R.keys()):
        log = R[name]
        ax.plot(_ex(log, "epoch"), _ex(log, "pred_seen_psnr"),
                color=_method_color(log), lw=1.4, label=_run_label(name, log), alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "plots", "all_psnr.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def plot_tau_trajectory(R, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("τ Trajectory (pcd runs)", fontsize=13, fontweight="bold")
    any_valid = False
    for name in sorted(R.keys()):
        log = R[name]
        eps = _ex(log, "epoch")
        taus = _ex(log, "tau")
        valid = [(e, t) for e, t in zip(eps, taus) if not math.isnan(t)]
        if valid:
            any_valid = True
            ax.plot([v[0] for v in valid], [v[1] for v in valid], lw=1.2, label=name, alpha=0.8)
    if not any_valid:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("τ")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "plots", "tau_trajectory.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ── Text summary ──────────────────────────────────────────────────────────────

def write_summary(R, out):
    lines = []
    lines.append("=" * 214)
    lines.append(
        f"{'Experiment':<44} {'Vid':<12} {'Method':<13} {'PSNRs':>7} {'PSNRu':>7} {'ChGrpS%':>8} {'ChGrpSparse/Tot':>16} "
        f"{'DH-S%':>7} {'ParamS%':>8} {'DecHead/Total params':>24} {'tau':>6} {'l_prox':>9} {'ws_w':>9} {'Seed':>5}"
    )
    lines.append("-" * 214)

    for name in sorted(R.keys()):
        f = R[name]["final"]
        cfg = R[name]["config"]
        tau = cfg.get("tau")
        lam = cfg.get("lambda_prox")
        ws = cfg.get("ws_weight")
        acct = cfg.get("accounting", {})
        dh = acct.get("decoder_head_params", 0)
        tot = acct.get("trainable_params", 0)
        psnr_u = f.get("pred_unseen_psnr", float("nan"))
        psnr_u_s = f"{psnr_u:>7.2f}" if isinstance(psnr_u, (int, float)) and not math.isnan(psnr_u) else "    N/A"

        lines.append(
            f"{name:<44} {cfg.get('vid', '?'):<12} {cfg.get('method', '?'):<13} "
            f"{f.get('pred_seen_psnr', 0):>7.2f} {psnr_u_s} "
            f"{f.get('group_sparsity', 0):>8.2f} "
            f"{str(f.get('group_sparse', 0)) + '/' + str(f.get('group_total', 0)):>16} "
            f"{f.get('decoder_head_param_sparsity', 0):>7.2f} "
            f"{f.get('model_param_sparsity', 0):>8.2f} "
            f"{str(dh) + '/' + str(tot):>24} "
            f"{(f'{tau:.3f}' if tau is not None else 'N/A'):>6} "
            f"{(f'{lam:.1e}' if lam is not None else 'N/A'):>9} "
            f"{(f'{ws:.1e}' if ws is not None else 'N/A'):>9} "
            f"{str(cfg.get('seed', '?')):>5}"
        )

    lines.append("=" * 214)
    txt = "\n".join(lines)
    print(f"\n{txt}\n")
    p = os.path.join(out, "summary.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(f"  Saved {p}")


# ── Top-level runner ──────────────────────────────────────────────────────────

def run_analysis(out):
    results_dir = os.path.join(out, "results")
    R = load_results(results_dir)
    if not R:
        print("No completed training results found.")
        return
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    print("\nGenerating plots...")
    plot_pareto(R, out)
    plot_training(R, out)
    plot_psnr_all(R, out)
    plot_tau_trajectory(R, out)
    write_summary(R, out)
