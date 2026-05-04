import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import MODEL_LABELS, plot_metric, run_model, write_csv


def summarize_final_blocks(rows):
    summary = []
    penalties = sorted({float(row["pu_penalty"]) for row in rows})
    for penalty in penalties:
        penalty_rows = [row for row in rows if float(row["pu_penalty"]) == penalty]
        final_block = max(int(row["block"]) for row in penalty_rows)
        for model in sorted({row["model"] for row in penalty_rows}):
            vals = [
                row for row in penalty_rows
                if row["model"] == model and int(row["block"]) == final_block
            ]
            if not vals:
                continue
            summary.append({
                "pu_penalty": penalty,
                "model": model,
                "avg_reward_mean": float(np.mean([float(row["avg_reward"]) for row in vals])),
                "avg_reward_std": float(np.std([float(row["avg_reward"]) for row in vals])),
                "success_rate_per_access_mean": float(np.mean([float(row["success_rate_per_access"]) for row in vals])),
                "pu_collision_rate_per_access_mean": float(np.mean([float(row["pu_collision_rate_per_access"]) for row in vals])),
                "su_collision_rate_per_access_mean": float(np.mean([float(row["su_collision_rate_per_access"]) for row in vals])),
                "access_rate_mean": float(np.mean([float(row["access_rate"]) for row in vals])),
                "avg_spikes_per_su_step_mean": float(np.mean([float(row["avg_spikes_per_su_step"]) for row in vals])),
            })
    return summary


def write_summary(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def is_combo(row, penalty, model_label, seed, env_seed):
    return (
        float(row["pu_penalty"]) == float(penalty)
        and row["model"] == model_label
        and int(row["seed"]) == int(seed)
        and int(row.get("env_seed", seed)) == int(env_seed)
    )


def plot_sweep(summary_rows, metric, ylabel, path):
    plt.figure(figsize=(9, 5))
    for model in sorted({row["model"] for row in summary_rows}):
        model_rows = sorted(
            [row for row in summary_rows if row["model"] == model],
            key=lambda row: float(row["pu_penalty"]),
        )
        xs = [float(row["pu_penalty"]) for row in model_rows]
        ys = [float(row[metric]) for row in model_rows]
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("PU collision penalty")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "dqn_rc",
        "actor_critic",
        "recurrent_actor_critic",
        "rstdp_snn",
    ])
    parser.add_argument("--penalties", type=float, nargs="+", default=[-2, -4, -6, -8, -10])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--block-size", type=int, default=2000)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--sus", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--env-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--sense-error", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dqn-learn-period", type=int, default=None)
    parser.add_argument("--dqn-memory-size", type=int, default=None)
    parser.add_argument("--dqn-replace-target-iter", type=int, default=1)
    parser.add_argument("--freeze-snn-on-converged", action="store_true")
    parser.add_argument("--freeze-success-rate", type=float, default=0.96)
    parser.add_argument("--freeze-pu-collision-rate", type=float, default=0.035)
    parser.add_argument("--freeze-su-collision-rate", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", default="result/penalty_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dqn_learn_period = args.dqn_learn_period or args.block_size
    dqn_memory_size = args.dqn_memory_size or args.block_size

    partial_path = out_dir / "penalty_sweep_block_metrics.partial.csv"
    all_rows = read_rows(partial_path) if args.resume else []
    expected_blocks = args.steps // args.block_size
    if args.resume and all_rows:
        print(f"Resuming from {partial_path} with {len(all_rows)} existing block rows")

    env_seeds = args.env_seeds if args.env_seeds is not None else [args.env_seed]
    if env_seeds == [None]:
        env_seeds = args.seeds

    for penalty in args.penalties:
        for env_seed in env_seeds:
            run_args = SimpleNamespace(
                models=args.models,
                steps=args.steps,
                block_size=args.block_size,
                channels=args.channels,
                sus=args.sus,
                seeds=args.seeds,
                env_seed=env_seed,
                sense_error=args.sense_error,
                pu_penalty=penalty,
                learning_rate=args.learning_rate,
                dqn_learn_period=dqn_learn_period,
                dqn_memory_size=dqn_memory_size,
                dqn_replace_target_iter=args.dqn_replace_target_iter,
                freeze_snn_on_converged=args.freeze_snn_on_converged,
                freeze_success_rate=args.freeze_success_rate,
                freeze_pu_collision_rate=args.freeze_pu_collision_rate,
                freeze_su_collision_rate=args.freeze_su_collision_rate,
                out_dir=str(out_dir),
            )
            for seed in args.seeds:
                for model in args.models:
                    model_label = MODEL_LABELS.get(model, model)
                    existing_combo_rows = [
                        row for row in all_rows
                        if is_combo(row, penalty, model_label, seed, env_seed)
                    ]
                    if len(existing_combo_rows) >= expected_blocks:
                        print(f"Penalty={penalty:g} env_seed={env_seed} skipping completed {model_label} seed={seed}")
                        continue
                    if existing_combo_rows:
                        print(f"Penalty={penalty:g} env_seed={env_seed} rerunning incomplete {model_label} seed={seed}")
                        all_rows = [
                            row for row in all_rows
                            if not is_combo(row, penalty, model_label, seed, env_seed)
                        ]
                    print(f"Penalty={penalty:g} env_seed={env_seed} running {MODEL_LABELS.get(model, model)} seed={seed}")
                    current_rows = []

                    def record_block(row):
                        row_with_penalty = dict(row)
                        row_with_penalty["pu_penalty"] = penalty
                        current_rows.append(row_with_penalty)
                        write_csv(all_rows + current_rows, out_dir / "penalty_sweep_block_metrics.partial.csv")
                        partial_summary = summarize_final_blocks(all_rows + current_rows)
                        write_summary(partial_summary, out_dir / "penalty_sweep_summary.partial.csv")
                        print(
                            f"  block={row['block']} step={row['step']} "
                            f"reward={row['avg_reward']:.3f} "
                            f"PUcoll/access={row['pu_collision_rate_per_access']:.3f} "
                            f"SUcoll/access={row['su_collision_rate_per_access']:.3f} "
                            f"frozen={row['learning_frozen']}",
                            flush=True,
                        )

                    run_model(model, run_args, seed, block_callback=record_block)
                    all_rows.extend(current_rows)

    write_csv(all_rows, out_dir / "penalty_sweep_block_metrics.csv")
    summary_rows = summarize_final_blocks(all_rows)
    write_summary(summary_rows, out_dir / "penalty_sweep_summary.csv")

    plot_sweep(
        summary_rows,
        "avg_reward_mean",
        "Final-block average reward",
        out_dir / "sweep_reward.png",
    )
    plot_sweep(
        summary_rows,
        "pu_collision_rate_per_access_mean",
        "Final-block PU collision rate per access",
        out_dir / "sweep_pu_collision_rate.png",
    )
    plot_sweep(
        summary_rows,
        "su_collision_rate_per_access_mean",
        "Final-block SU collision rate per access",
        out_dir / "sweep_su_collision_rate.png",
    )
    plot_sweep(
        summary_rows,
        "access_rate_mean",
        "Final-block access rate",
        out_dir / "sweep_access_rate.png",
    )

    # Also save training curves collapsed across penalties into separate files.
    for penalty in args.penalties:
        rows = [row for row in all_rows if float(row["pu_penalty"]) == float(penalty)]
        safe_penalty = str(penalty).replace("-", "neg").replace(".", "p")
        plot_metric(
            rows,
            "avg_reward",
            "Average reward per SU per step",
            out_dir / f"training_reward_penalty_{safe_penalty}.png",
        )
        plot_metric(
            rows,
            "pu_collision_rate_per_access",
            "PU collision rate per access",
            out_dir / f"training_pu_collision_penalty_{safe_penalty}.png",
        )

    print("\nFinal-block penalty sweep summary")
    for row in summary_rows:
        print(
            f"penalty={row['pu_penalty']:>5g} {row['model']}: "
            f"reward={row['avg_reward_mean']:.3f}, "
            f"PUcoll/access={row['pu_collision_rate_per_access_mean']:.3f}, "
            f"SUcoll/access={row['su_collision_rate_per_access_mean']:.3f}, "
            f"access={row['access_rate_mean']:.3f}"
        )
    print(f"\nSaved sweep results to {out_dir}")


if __name__ == "__main__":
    main()
