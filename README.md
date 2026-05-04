# Spiking Actor-Critic for Dynamic Spectrum Access

This repository contains a Dynamic Spectrum Access (DSA) reinforcement learning project comparing ANN and SNN actor-critic methods under noisy channel sensing.

The project is based on the DSA environment style from:

> H.-H. Chang, H. Song, Y. Yi, J. Zhang, H. He, and L. Liu, "Distributive Dynamic Spectrum Access Through Deep Reinforcement Learning: A Reservoir Computing-Based Approach," IEEE Internet of Things Journal, 2019.

# LLM Use Acknowledgement

Portions of this code were generated using ChatGPT/Codex. All portions of code were reviewed by auther before inclusions


## Repository Contents

- `DSA_env.py` - DSA Markov environment with noisy sensing, PU/SU interference, SINR rewards, and collision metrics.
- `run_experiments.py` - Main experiment runner for individual model comparisons.
- `sweep_penalties.py` - Penalty sweep runner with resumable partial CSV logging.
- `OnlineActorCritic.py` - Feedforward ANN actor-critic baseline.
- `RecurrentActorCritic.py` - Recurrent ANN actor-critic baseline.
- `OnlineActorCriticRSTDP.py` - Reward-modulated STDP-style SNN actor-critic.
- `OnlineActorCriticLSM.py` - Recurrent / liquid-state SNN actor-critic.
- `OnlineActorCriticSNNPop.py` - Earlier population-coded SNN actor-critic baseline.
- `DQN_RC.py` and `pyESN_online.py` - Reservoir-computing DQN code adapted from the reference project.
- `Run_Models.ipynb` - Clean example notebook for code submission.
- `environmetn_analysis.ipynb` - notebook showing ablation studies on the environment
- `result/` - Saved CSV metrics and generated plots from completed/partial runs.


## Setup

The minimal requirements are listed in `requirements.txt`.

## Example Notebook

Use `Run_Models.ipynb` as the code-submission example notebook. It runs the final-style fixed-environment experiment, writes CSV outputs, saves plots, and displays the plots inline. Outputs are saved to `result/notebook_final_experiment`.

`main.ipynb` is retained as an older exploratory notebook, but it is not the recommended submission demo.

## Running Experiments

Quick smoke run:


python run_experiments.py --steps 5000 --block-size 1000 --models safe_sensed myopic_noisy actor_critic recurrent_actor_critic rstdp_snn lsm_snn --seeds 0 --env-seed 0 --pu-penalty -6 --freeze-snn-on-converged --out-dir result/smoke_submission_check


Final fixed-environment run used for the main report table:


python sweep_penalties.py --steps 100000 --block-size 2000 --models recurrent_actor_critic rstdp_snn lsm_snn --penalties -6 --seeds 0 1 2 --env-seed 0 --freeze-snn-on-converged --out-dir result/final_penalty_neg6_100k_freeze_env0_3seed


Larger multi-penalty sweep, resumable:


python sweep_penalties.py --steps 100000 --block-size 2000 --models safe_sensed myopic_noisy actor_critic recurrent_actor_critic rstdp_snn lsm_snn --penalties -2 -4 -5 -6 --seeds 0 --env-seeds 0 1 2 --freeze-snn-on-converged --resume --out-dir result/penalty_sweep_100k_envseeds_0_1_2


`sweep_penalties.py` writes partial CSV files after every block, so interrupted long runs can still be analyzed or resumed.

## Outputs

`run_experiments.py` writes:

- `block_metrics.csv`
- `summary_final_block.csv`
- reward/collision plots

`sweep_penalties.py` writes:

- `penalty_sweep_block_metrics.csv`
- `penalty_sweep_summary.csv`
- `penalty_sweep_block_metrics.partial.csv`
- `penalty_sweep_summary.partial.csv`
- sweep and training plots

## Model Checkpoints

The models are trained online inside each experiment run. and the point of this project is online/reinforecement learning. As such it doesn't make sense to have pretrained models.
To reproduce the reported trained behavior, rerun the commands above with the same:

- `--env-seed`
- `--seeds`
- `--pu-penalty`
- `--steps`
- `--block-size`
- model list
- freeze settings


## Notes

- `safe_sensed` is the fairest non-learning baseline because it only uses the same noisy sensing vector given to the learning agents.
- `myopic_noisy` uses noisy sensing plus known transition/sensing statistics.
- `myopic` / `myopic_oracle` is retained only as a diagnostic because it uses a true-state-dependent SINR table.
- `--freeze-snn-on-converged` freezes SNN plasticity after a block reaches high success and low collision rates.
