# DQN Atari Assignment - Boxing 

## Group 4 Members:

- Elyse Marie Uyiringiye
- Nice Eva Karabaranga
- Best Verie Iradukunda
- Raissa Irutingabo

## Model Files/Videos 

[https://drive.google.com/drive/folders/18KEPQ7AlXoZxDiDu8lFAwod63G7g-8rQ?usp=sharing]
## Best Model Playing: 

[Best Model](https://drive.google.com/file/d/1rm9EKlDZ269LD7_plWfo2yjR2vQQzB_5/view?usp=drive_link)

## Overview

This project implements a Deep Q-Network (DQN) agent trained to play the Atari Boxing-v5 environment using Stable Baselines3 and Gymnasium. This requires training and comparing multiple DQN configurations with different hyperparameters and policy architectures (CNNPolicy vs MLPPolicy).


## Environment
- Gymnasium Atari environment: `ALE/Boxing-v5`
- Framework: Stable-Baselines3 `DQN`

## Scripts
- `scripts/train.py`: trains different DQN experiments  and saves model artifacts
- `scripts/play.py`: loads trained best model and runs evaluation gameplay


## Key Libraries

stable-baselines3 (≥2.3.2): DQN algorithm and policy implementations
gymnasium[atari] (≥1.0.0): Atari environment wrapper
ale-py (≥0.10.1): Atari Learning Environment interface
torch: Neural network backend
pandas, matplotlib: Data analysis and visualization

## Installation

# Install dependencies
`pip install -r requirements.txt`

# On some systems, may need to install ROMs separately
`AutoROM --accept-license`

# Group Members Contribution

## Best Verie Experiments


## Best Verie Iradukunda's Experiment Summary (11 experiments)

| Experiment | Policy | Mean Reward | Std Reward | Train Time (min) | Notes |
|-----------|--------|------------|------------|------------------|------|
| boxing_exp01_baseline_cnn | CnnPolicy | **5.9** | 5.50 | 14.23 |  Best overall |
| boxing_exp05_high_gamma_cnn | CnnPolicy | 4.7 | 3.66 | 14.11 | Higher gamma |
| boxing_exp08_less_exploration_cnn | CnnPolicy | 4.1 | 4.50 | 14.36 | Less exploration |
| boxing_exp02_small_batch_cnn | CnnPolicy | 3.8 | 5.27 | 13.62 | Smaller batch |
| boxing_exp04_low_gamma_cnn | CnnPolicy | 1.2 | 3.91 | 14.08 | Lower gamma |
| boxing_exp06_gamma_zero_cnn | CnnPolicy | -0.3 | 6.33 | 14.16 | No future reward |
| boxing_exp03_large_batch_cnn | CnnPolicy | -0.6 | 3.29 | 14.75 | Large batch |
| boxing_exp07_more_exploration_cnn | CnnPolicy | -0.7 | 5.08 | 13.96 | More exploration |
| boxing_exp09_more_updates_cnn | CnnPolicy | -1.3 | 4.27 | 19.49 | More gradient steps |
| boxing_exp11_small_batch_mlp | MlpPolicy | -14.4 | 5.64 | 7.04 | MLP policy |
| boxing_exp10_baseline_mlp | MlpPolicy | -27.3 | 4.47 | 7.18 | Worst performance |


##  Best Model

The best-performing model is:boxing_exp01_baseline_cnn with mean reward of 5.9

The baseline CNN model outperformed others because it used balanced default hyperparameters that ensured stable learning. With only 100,000 timesteps, modified settings like high gamma or altered exploration failed to converge. The CNN also captured spatial features effectively, making it more robust than other configurations, especially under limited training conditions.
## Key files

###  experiments/BestVerie_experiments.ipynb: Full training pipeline with callbacks and visualization

- Hyperparameter_tables/BestVerie_hyperparameter_results.csv: Summary results table
- In drive link: /BestVerie.zip: logs,models, everything related to the training process

## Raissa IRUTINAGBO's experiments results


Raissa Experiments (10 Configurations)Notebook used: experiments/irutingabo-experiments.ipynbThis section defines 10 experiment configurations arranged as 5 paired CNN/MLP groups — each pair shares identical hyperparameters so that CnnPolicy and MlpPolicy can be compared fairly under the same conditions.CSV Results: results/raissa/tables/raissa_hyperparameter_results.csvMarkdown Results: results/raissa/tables/raissa_hyperparameter_results.md

## Hyperparamter experiments

| Member Name | Experiment                   | Hyperparameter Set                                                                         | Noted Behavior                                                                                                              |
|-------------|------------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Raissa      | raissa_exp01_cnn_baseline    | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1   | CNN configuration for Boxing; compare stability and final reward against baseline. Mean reward: -9.80 +/- 9.41.            |
| Raissa      | raissa_exp02_mlp_baseline    | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1   | MLP ablation on Boxing; typically weaker than CNN on pixel observations. Mean reward: -6.80 +/- 6.85.                      |
| Raissa      | raissa_exp03_cnn_low_lr      | lr=5e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | CNN configuration for Boxing; compare stability and final reward against baseline. Mean reward: -1.20 +/- 1.17.            |
| Raissa      | raissa_exp04_mlp_low_lr      | lr=5e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | MLP ablation on Boxing; typically weaker than CNN on pixel observations. Mean reward: -2.40 +/- 6.34.                      |
| Raissa      | raissa_exp05_cnn_high_lr     | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1  | CNN configuration for Boxing; compare stability and final reward against baseline. Mean reward: -7.20 +/- 8.01.            |
| Raissa      | raissa_exp06_mlp_high_lr     | lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1  | MLP ablation on Boxing; typically weaker than CNN on pixel observations. Mean reward: -7.00 +/- 4.94.                      |
| Raissa      | raissa_exp07_cnn_slow_eps    | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5    | CNN configuration for Boxing; compare stability and final reward against baseline. Mean reward: -8.00 +/- 9.90.            |
| Raissa      | raissa_exp08_mlp_slow_eps    | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.5    | MLP ablation on Boxing; typically weaker than CNN on pixel observations. Mean reward: -15.00 +/- 3.74.                     |
| Raissa      | raissa_exp09_cnn_large_batch | lr=0.0001, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1  | CNN configuration for Boxing; compare stability and final reward against baseline. Mean reward: -4.00 +/- 6.20.            |
| Raissa      | raissa_exp10_mlp_large_batch | lr=0.0001, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1  | MLP ablation on Boxing; typically weaker than CNN on pixel observations. Mean reward: 2.80 +/- 5.46.                       |


## Results

| Experiment Name              | Policy    | Mean Reward | Std Dev | Train Min |
|------------------------------|-----------|-------------|---------|-----------|
| raissa_exp10_mlp_large_batch | MlpPolicy |  2.8        | 5.46    | 1.53      |
| raissa_exp03_cnn_low_lr      | CnnPolicy | -1.2        | 1.17    | 5.08      |
| raissa_exp04_mlp_low_lr      | MlpPolicy | -2.4        | 6.34    | 1.55      |
| raissa_exp09_cnn_large_batch | CnnPolicy | -4.0        | 6.20    | 6.34      |
| raissa_exp02_mlp_baseline    | MlpPolicy | -6.8        | 6.85    | 1.53      |
| raissa_exp06_mlp_high_lr     | MlpPolicy | -7.0        | 4.94    | 1.53      |
| raissa_exp05_cnn_high_lr     | CnnPolicy | -7.2        | 8.01    | 5.04      |
| raissa_exp07_cnn_slow_eps    | CnnPolicy | -8.0        | 9.90    | 4.82      |
| raissa_exp01_cnn_baseline    | CnnPolicy | -9.8        | 9.41    | 5.13      |
| raissa_exp08_mlp_slow_eps    | MlpPolicy | -15.0       | 3.74    | 1.46      |


Findings and Insights

Best model: Raissa Exp10 MLP Large Batch, with a mean reward of 2.8. It was the only experiment to achieve a positive score across all 10 runs.

Surprising result: MLP outperformed CNN overall. The average mean reward across all five CNN experiments was -6.04, while the MLP average was -5.68. This is unexpected since Boxing is a pixel-based environment. However, with only 50,000 timesteps, the CNN likely did not have enough time to learn useful visual features. In contrast, the MLP trained faster (about 1.5 minutes vs 5 minutes) and made better use of the limited training budget.

What helped: Lower learning rate. Reducing the learning rate to 5e-5 produced the best CNN result (-1.2) and the second-best MLP result (-2.4). More cautious updates appear to stabilize training under a short timestep budget.

What helped: Larger batch size for MLP. The MLP with a batch size of 128 performed best overall. Smoother gradient estimates provided a stronger learning signal, allowing the model to develop a basic positive strategy.

What hurt: Higher learning rate. At 2.5e-4, updates were too aggressive, leading to unstable training and inconsistent Q-value estimates.

What hurt the most: Slow epsilon decay. Extending exploration to 50% of training was the worst-performing decision. Spending too much of a short 50k-step budget on random exploration left insufficient time for effective exploitation.

## MarieAElyse — ALE/Boxing-v5

### Environment
| Property | Value |
|----------|-------|
| Environment | `ALE/Boxing-v5` |
| Action Space | `Discrete(18)` |
| Observation Space | `Box(0, 255, (210, 160, 3), uint8)` |
| Algorithm | DQN + CnnPolicy |
| Total Timesteps | 100,000 per experiment |

---

### Hyperparameter Table

| Member | Hyperparameter Set | Noted Behavior |
|--------|-------------------|----------------|
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 1 - Baseline] Stable training. Reward improves steadily. Agent learns to land punches over time. Reference point for all other experiments. Mean Reward: 3.8 |
| Elyse | lr=5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 2 - High LR] Q-values diverge catastrophically. Training loss spikes. lr=5e-4 is too aggressive — worst experiment. Mean Reward: -41.0 |
| Elyse | lr=1e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 3 - Low LR] Very slow convergence. Agent still near-random at midpoint. Rewards far below baseline. Mean Reward: -4.4 |
| Elyse | lr=1e-4, gamma=0.90, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 4 - Low Gamma] Agent is myopic — ignores long-term scoring. Rewards plateau lower than baseline. Mean Reward: 1.4 |
| Elyse | lr=1e-4, gamma=0.999, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 5 - High Gamma] Agent values long-term strategy. Slightly slower early learning but more deliberate play. Mean Reward: 3.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 6 - Large Batch] Smoother loss curve but fewer updates per timestep. Final performance below baseline. Mean Reward: -1.2 |
| Elyse | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 | [Exp 7 - Small Batch] Noisy gradients but frequent updates. Surprisingly good performance. Mean Reward: 7.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay=0.50 | [Exp 8 - Slow Epsilon Decay] Extended exploration fills replay buffer with diverse transitions. Mean Reward: 3.4 |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05 | [Exp 9 - Fast Epsilon Decay] Commits to exploitation early. Best result — Boxing is simple enough that fast exploitation beats extended exploration. **Mean Reward: 11.2 ✅ Best** |
| Elyse | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.10 [MlpPolicy] | [Exp 10 - MLP Ablation] Same hyperparameters as Exp 1, only policy differs. Cannot extract spatial features. 37-point gap vs CNN confirms CnnPolicy is essential. Mean Reward: -33.2 |

---

### Best Model — exp09_fast_eps

| Metric | Value |
|--------|-------|
| Mean Reward (training eval) | 11.2 ± 6.76 |
| Mean Reward (live play) | 8.6 ± 1.85 |
| Win Rate | 5W / 0D / 0L (100%) |
| Best Episode | 11.0 |
| Worst Episode | 6.0 |

---

### Key Insights

- **Best config:** `exp09_fast_eps` — fast epsilon decay (5% of steps) worked best for Boxing because the environment is simple enough to benefit from early exploitation
- **Worst config:** `exp02_high_lr` — lr=5e-4 caused Q-value divergence, scoring -41.0
- **CNN vs MLP:** CnnPolicy scored 3.8 vs MlpPolicy -33.2 with identical hyperparameters — a 37-point gap proving CNN is essential for pixel-based Atari
- **Surprise finding:** Fast epsilon decay outperformed slow decay, contrary to theory — Boxing's dense reward signal allows the agent to learn a good policy quickly

---

### Gameplay Video

>  [Watch the agent play Boxing](videos/boxing_gameplay.avi)

Agent uses **GreedyQPolicy** (`exploration_rate=0.0` → always picks `argmax Q(s,a)`).

## Nice Eva Karabaranga - ALE/Boxing-v5

### Environment
| Property | Value |
|----------|-------|
| Environment | `ALE/Boxing-v5` |
| Action Space | `Discrete(18)` |
| Observation Space | `Box(0, 255, (210, 160, 3), uint8)` |
| Algorithm | DQN + CnnPolicy |
| Total Timesteps | 100,000 per experiment |

---

### Hyperparameter Table

| Member | Hyperparameter Set | Noted Behavior |
|--------|-------------------|----------------|
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 1 - Baseline] Moderate config, stable reference point. Mean Reward: -0.20 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 2 - Zero Eps End] Fully greedy at end, there's no residual exploration. **Best performer. Mean Reward: +4.40 ✅ Best** |
| Nice | lr=2.5e-4, gamma=0.0, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 3 - Zero Gamma] Fully myopic agent ignores all future rewards. Expected poor performance confirmed. Mean Reward: -2.80 |
| Nice | lr=1e-3, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 4 - Very High LR] Aggressive updates cause unstable Q-values. Worst CNN experiment. Mean Reward: -24.80 |
| Nice | lr=1e-6, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 5 - Tiny LR] Near-zero learning rate: agent barely updates. Stagnant performance. Mean Reward: -11.80 |
| Nice | lr=2.5e-4, gamma=0.999, batch=256, eps_start=1.0, eps_end=0.02, eps_fraction=0.20 | [Exp 6 - Large Batch + High Gamma] Stable gradients with strong future valuation. Mean Reward: +0.40 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.01 | [Exp 7 - Instant Exploit] Epsilon collapses in the first 1% of training; almost no exploration phase. Mean Reward: +2.00 |
| Nice | lr=2.5e-4, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.02, eps_fraction=0.90 | [Exp 8 - Full Explore] Explores for 90% of training: agent learns slowly and struggles to exploit. Mean Reward: 0.00 |
| Nice | lr=2.5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.01, eps_fraction=0.15 | [Exp 9 - Best Guess] Balanced config combining good gamma and low eps_end. Mean Reward: +1.00 |
| Nice | lr=2.5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 11 - Zero Eps + High Gamma] Greedy convergence combined with strong future valuation. Mean Reward: -0.20 |
| Nice | lr=2.5e-4, gamma=0.95, batch=128, eps_start=1.0, eps_end=0.0, eps_fraction=0.20 | [Exp 12 - Zero Eps + Large Batch] Larger batch stabilises gradients with fully greedy end. Mean Reward: -0.40 |
| Nice | lr=5e-4, gamma=0.99, batch=64, eps_start=1.0, eps_end=0.0, eps_fraction=0.15 | [Exp 13 - Tuned LR + Zero Eps] Slightly higher LR with zero eps and high gamma; LR too high for zero eps. Mean Reward: -20.20 |

---

### Best Model: exp02_zero_eps_end

| Metric | Value |
|--------|-------|
| Mean Reward | 4.40 ± 3.72 |
| Policy | CnnPolicy |
| Key Setting | eps_end=0.0 (fully greedy convergence) |

---

### Key Insights

- **Best config:** `exp02_zero_eps_end` :setting epsilon_end to 0.0 forces fully greedy exploitation at the end of training, giving the highest reward
- **Worst config:** `exp04_very_high_lr` : lr=1e-3 caused Q-value instability, scoring -24.80
- **Zero gamma finding:** Setting gamma=0.0 makes the agent fully myopic, meaning it only optimises for immediate reward and cannot learn long-term boxing strategy
- **Exploration tradeoff:** Both instant exploitation (exp07) and full exploration (exp08) underperformed, confirming that a balanced epsilon decay is important

---

### Artifacts
- Notebook: `experiments/Nice_experiments.ipynb`
- Best model: `Nice_dqn_model.zip`
- Hyperparameter table: `hyperparameter_table_nice.csv`
- Reward comparison: `assets/nice_reward_comparison.png`
- Training curves: `assets/nice_training_curves.png`

## Demo: 
[My best model](https://drive.google.com/file/d/1YmAl1JB5iatSVjZoA1seLZj41rOzOwoL/view?usp=drive_link) 
