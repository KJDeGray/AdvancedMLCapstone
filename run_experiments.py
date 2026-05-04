import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from DSA_env import DSA_Markov
from DQN_RC import DeepQNetwork
from OnlineActorCritic import OnlineActorCritic
from OnlineActorCriticLSM import OnlineActorCriticLSM
from OnlineActorCriticRSTDP import OnlineActorCriticRSTDP
from OnlineActorCriticSNNPop import OnlineActorCriticSNNPop
from RecurrentActorCritic import RecurrentActorCritic


MODEL_LABELS = {
    "myopic": "Myopic",
    "myopic_oracle": "OracleMyopic",
    "myopic_noisy": "NoisyMyopic",
    "safe_sensed": "SafeSensed",
    "dqn_rc": "DQN-RC",
    "actor_critic": "OnlineActorCritic",
    "recurrent_actor_critic": "RecurrentActorCritic",
    "snn_pop": "OnlineActorCriticSNNPop",
    "rstdp_snn": "OnlineActorCriticRSTDP",
    "lsm_snn": "OnlineActorCriticLSM",
}


def _myopic_idle_probability(env, observation, su):
    return (
        observation[su, :] * (1 - env.sense_error_prob[su, :])
        + (1 - observation[su, :]) * env.sense_error_prob[su, :]
    )


def choose_myopic_actions(env, observation):
    action = np.zeros(env.n_su, dtype=np.int32)
    for su in range(env.n_su):
        idle_prob = _myopic_idle_probability(env, observation, su)
        expected_reward = (
            idle_prob * (env.goodToBad_prob * env.punish_interfer_PU)
            + idle_prob * (env.stayGood_prob * np.log2(1 + env.SINR[su, :]))
            + (1 - idle_prob) * (env.stayBad_prob * env.punish_interfer_PU)
            + (1 - idle_prob) * (env.badToGood_prob * np.log2(1 + env.SINR[su, :]))
        )
        action[su] = int(np.argmax(expected_reward)) if np.max(expected_reward) > 0 else env.n_channel
    return action


def choose_noisy_myopic_actions(env, observation):
    action = np.zeros(env.n_su, dtype=np.int32)
    no_pu_rate = np.log2(1 + env.H2 * env.SU_power / env.Noise)
    for su in range(env.n_su):
        idle_prob = _myopic_idle_probability(env, observation, su)
        expected_idle_next = idle_prob * env.stayGood_prob + (1 - idle_prob) * env.badToGood_prob
        expected_reward = (
            expected_idle_next * no_pu_rate[su, :]
            + (1 - expected_idle_next) * env.punish_interfer_PU
        )
        action[su] = int(np.argmax(expected_reward)) if np.max(expected_reward) > 0 else env.n_channel
    return action


def choose_safe_sensed_actions(env, observation):
    action = np.full(env.n_su, env.n_channel, dtype=np.int32)
    for su in range(env.n_su):
        safe_channels = np.flatnonzero(observation[su, :] == 1)
        if len(safe_channels) > 0:
            action[su] = int(safe_channels[0])
    return action


def make_agents(model_name, env, seed, learning_rate):
    agents = []
    for su in range(env.n_su):
        agent_seed = seed + 1000 * (su + 1)
        if model_name == "dqn_rc":
            memory_size = getattr(env, "_runner_dqn_memory_size", 300)
            replace_target_iter = getattr(env, "_runner_dqn_replace_target_iter", 1)
            agents.append(
                DeepQNetwork(
                    env.n_actions,
                    env.n_features,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    replace_target_iter=replace_target_iter,
                    memory_size=memory_size,
                    lr=learning_rate,
                )
            )
        elif model_name == "actor_critic":
            agents.append(
                OnlineActorCritic(
                    env.n_actions,
                    env.n_features,
                    learning_rate=learning_rate,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    seed=agent_seed,
                )
            )
        elif model_name == "recurrent_actor_critic":
            agents.append(
                RecurrentActorCritic(
                    env.n_actions,
                    env.n_features,
                    learning_rate=learning_rate,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    hidden_size=48,
                    recurrent_scale=0.8,
                    seed=agent_seed,
                )
            )
        elif model_name == "snn_pop":
            agents.append(
                OnlineActorCriticSNNPop(
                    env.n_actions,
                    env.n_features,
                    learning_rate=learning_rate,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    hidden_size=32,
                    actor_pop_size=4,
                    critic_pop_size=8,
                    spike_window=30,
                    seed=agent_seed,
                )
            )
        elif model_name == "rstdp_snn":
            agents.append(
                OnlineActorCriticRSTDP(
                    env.n_actions,
                    env.n_features,
                    learning_rate=learning_rate,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    hidden_size=48,
                    actor_pop_size=6,
                    critic_pop_size=8,
                    spike_window=30,
                    seed=agent_seed,
                )
            )
        elif model_name == "lsm_snn":
            agents.append(
                OnlineActorCriticLSM(
                    env.n_actions,
                    env.n_features,
                    learning_rate=learning_rate,
                    reward_decay=0.9,
                    e_greedy=0.8,
                    hidden_size=64,
                    actor_pop_size=6,
                    critic_pop_size=8,
                    spike_window=30,
                    seed=agent_seed,
                )
            )
        else:
            raise ValueError(f"Unknown agent model: {model_name}")
    return agents


def set_exploit_probability(agents, step, total_steps):
    if not agents:
        return
    if step < total_steps * 0.25:
        epsilon = 0.8
    elif step < total_steps * 0.5:
        epsilon = 0.9
    else:
        epsilon = 1.0
    for agent in agents:
        if hasattr(agent, "epsilon"):
            agent.epsilon = epsilon


def maybe_freeze_snn_agents(model_name, agents, args, row):
    if not args.freeze_snn_on_converged:
        return False
    if model_name not in {"rstdp_snn", "lsm_snn", "snn_pop"}:
        return False
    if not agents or not all(hasattr(agent, "freeze_learning") for agent in agents):
        return False
    if any(getattr(agent, "learning_frozen", False) for agent in agents):
        return True

    safe_enough = (
        row["success_rate_per_access"] >= args.freeze_success_rate
        and row["pu_collision_rate_per_access"] <= args.freeze_pu_collision_rate
        and row["su_collision_rate_per_access"] <= args.freeze_su_collision_rate
    )
    if safe_enough:
        for agent in agents:
            agent.freeze_learning()
        return True
    return False


def run_model(model_name, args, seed, block_callback=None):
    env_seed = seed if getattr(args, "env_seed", None) is None else int(args.env_seed)
    np.random.seed(env_seed)
    env = DSA_Markov(
        n_channel=args.channels,
        n_su=args.sus,
        sense_error_prob_max=args.sense_error,
        punish_interfer_PU=args.pu_penalty,
        plot_locations=False,
    )
    env._runner_dqn_memory_size = args.dqn_memory_size or args.block_size
    env._runner_dqn_replace_target_iter = args.dqn_replace_target_iter

    baseline_models = {"myopic", "myopic_oracle", "myopic_noisy", "safe_sensed"}
    agents = [] if model_name in baseline_models else make_agents(model_name, env, seed, args.learning_rate)
    observation = env.sense()

    block_rows = []
    reward_sum = np.zeros(env.n_su)
    success_sum = 0
    fail_pu_sum = 0
    fail_collision_sum = 0
    spike_count_sum = 0.0
    access_sum = 0

    for step in range(args.steps):
        set_exploit_probability(agents, step, args.steps)

        if model_name in {"myopic", "myopic_oracle"}:
            action = choose_myopic_actions(env, observation)
        elif model_name == "myopic_noisy":
            action = choose_noisy_myopic_actions(env, observation)
        elif model_name == "safe_sensed":
            action = choose_safe_sensed_actions(env, observation)
        else:
            action = np.zeros(env.n_su, dtype=np.int32)
            for su, agent in enumerate(agents):
                action[su] = agent.choose_action(observation[su, :])
                if hasattr(agent, "last_spike_count"):
                    spike_count_sum += agent.last_spike_count()

        reward = env.access(action)
        env.render()
        env.render_SINR()
        next_observation = env.sense()

        if model_name == "dqn_rc":
            for su, agent in enumerate(agents):
                agent.store_transition(observation[su, :], action[su], reward[su], next_observation[su, :])
            if (step + 1) % args.dqn_learn_period == 0:
                for agent in agents:
                    agent.learn()
        elif model_name not in baseline_models:
            for su, agent in enumerate(agents):
                agent.learn(observation[su, :], action[su], reward[su], next_observation[su, :])
                if hasattr(agent, "spike_window") and not hasattr(agent, "last_spike_count"):
                    spike_count_sum += float(agent.spike_window)

        reward_sum += reward
        success_sum += env.success
        fail_pu_sum += env.fail_PU
        fail_collision_sum += env.fail_collision
        access_sum += int(np.sum(action != env.n_channel))
        observation = next_observation

        if (step + 1) % args.block_size == 0:
            denom = args.block_size * env.n_su
            access_denom = max(access_sum, 1)
            row = {
                "model": MODEL_LABELS[model_name],
                "seed": seed,
                "env_seed": env_seed,
                "block": (step + 1) // args.block_size,
                "step": step + 1,
                "avg_reward": float(np.sum(reward_sum) / denom),
                "success_per_su": float(success_sum / env.n_su),
                "pu_collision_per_su": float(fail_pu_sum / env.n_su),
                "su_collision_per_su": float(fail_collision_sum / env.n_su),
                "access_rate": float(access_sum / denom),
                "success_rate_per_access": float(success_sum / access_denom),
                "pu_collision_rate_per_access": float(fail_pu_sum / access_denom),
                "su_collision_rate_per_access": float(fail_collision_sum / access_denom),
                "avg_spikes_per_su_step": float(spike_count_sum / denom),
                "learning_frozen": False,
            }
            row["learning_frozen"] = maybe_freeze_snn_agents(model_name, agents, args, row)
            block_rows.append(row)
            if block_callback is not None:
                block_callback(row)
            reward_sum[:] = 0
            success_sum = 0
            fail_pu_sum = 0
            fail_collision_sum = 0
            spike_count_sum = 0.0
            access_sum = 0

    return block_rows


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, metric, ylabel, path):
    plt.figure(figsize=(9, 5))
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        steps = sorted({int(row["step"]) for row in model_rows})
        means = []
        for step in steps:
            vals = [float(row[metric]) for row in model_rows if int(row["step"]) == step]
            means.append(float(np.mean(vals)))
        plt.plot(steps, means, label=model)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def summarize(rows):
    final_block = max(int(row["block"]) for row in rows)
    lines = []
    for model in sorted({row["model"] for row in rows}):
        vals = [row for row in rows if row["model"] == model and int(row["block"]) == final_block]
        lines.append({
            "model": model,
            "avg_reward": float(np.mean([float(row["avg_reward"]) for row in vals])),
            "success_per_su": float(np.mean([float(row["success_per_su"]) for row in vals])),
            "pu_collision_per_su": float(np.mean([float(row["pu_collision_per_su"]) for row in vals])),
            "su_collision_per_su": float(np.mean([float(row["su_collision_per_su"]) for row in vals])),
            "access_rate": float(np.mean([float(row["access_rate"]) for row in vals])),
            "pu_collision_rate_per_access": float(np.mean([float(row["pu_collision_rate_per_access"]) for row in vals])),
            "success_rate_per_access": float(np.mean([float(row["success_rate_per_access"]) for row in vals])),
            "avg_spikes_per_su_step": float(np.mean([float(row["avg_spikes_per_su_step"]) for row in vals])),
        })
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["myopic", "actor_critic", "recurrent_actor_critic", "snn_pop", "rstdp_snn", "lsm_snn"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--sus", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--sense-error", type=float, default=0.2)
    parser.add_argument("--pu-penalty", type=float, default=-2.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dqn-learn-period", type=int, default=None)
    parser.add_argument("--dqn-memory-size", type=int, default=None)
    parser.add_argument("--dqn-replace-target-iter", type=int, default=1)
    parser.add_argument("--freeze-snn-on-converged", action="store_true")
    parser.add_argument("--freeze-success-rate", type=float, default=0.96)
    parser.add_argument("--freeze-pu-collision-rate", type=float, default=0.035)
    parser.add_argument("--freeze-su-collision-rate", type=float, default=0.005)
    parser.add_argument("--out-dir", default="result/clean_runs")
    args = parser.parse_args()
    if args.dqn_learn_period is None:
        args.dqn_learn_period = args.block_size

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        for model in args.models:
            print(f"Running {MODEL_LABELS.get(model, model)} seed={seed}")
            all_rows.extend(run_model(model, args, seed))

    write_csv(all_rows, out_dir / "block_metrics.csv")
    summary_rows = summarize(all_rows)
    write_csv(summary_rows, out_dir / "summary_final_block.csv")

    plot_metric(all_rows, "avg_reward", "Average reward per SU per step", out_dir / "avg_reward.png")
    plot_metric(all_rows, "success_per_su", "Successes per SU per block", out_dir / "success.png")
    plot_metric(all_rows, "pu_collision_per_su", "PU collisions per SU per block", out_dir / "pu_collision.png")
    plot_metric(all_rows, "pu_collision_rate_per_access", "PU collision rate per access", out_dir / "pu_collision_rate.png")
    plot_metric(all_rows, "su_collision_per_su", "SU collisions per SU per block", out_dir / "su_collision.png")

    print("\nFinal block summary")
    for row in summary_rows:
        print(
            f"{row['model']}: reward={row['avg_reward']:.3f}, "
            f"success/SU={row['success_per_su']:.1f}, "
            f"PUcoll/SU={row['pu_collision_per_su']:.1f}, "
            f"PUcoll/access={row['pu_collision_rate_per_access']:.3f}, "
            f"access_rate={row['access_rate']:.3f}, "
            f"SUcoll/SU={row['su_collision_per_su']:.1f}, "
            f"spikes/SUstep={row['avg_spikes_per_su_step']:.1f}"
        )
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
