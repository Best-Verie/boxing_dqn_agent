# DQN Atari Assignment - Boxing 

## Group 3 Members:

Elyse Marie Uyiringiye
Nice Eva
Best Verie Iradukunda
Raissa Irutingabo

## Model Files/Videos 

[!https://drive.google.com/drive/folders/18KEPQ7AlXoZxDiDu8lFAwod63G7g-8rQ?usp=sharing]
## Best Model Playing: 
### (paste the Best model link video here )


## Overview

This project implements a Deep Q-Network (DQN) agent trained to play the Atari Boxing-v5 environment using Stable Baselines3 and Gymnasium. This requires training and comparing multiple DQN configurations with different hyperparameters and policy architectures (CNNPolicy vs MLPPolicy).


## Environment
- Gymnasium Atari environment: `ALE/Tennis-v5`
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
pip install -r requirements.txt

# On some systems, may need to install ROMs separately
AutoROM --accept-license

# Group Members Contribution

## Best Verie Experiments

# Boxing DQN Experiments (ALE/Boxing-v5)


## Best Verie's Experiment Summary (11 experiments)

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
Files:
## Key files

experiments/BestVerie_experiments.ipynb: Full training pipeline with callbacks and visualization
Hyperparameter_tables/BestVerie_hyperparameter_results.csv: Summary results table
results/BestVerie.zip: logs,models, everything related to the training process
Video Demonstration:

## Demo

[![Watch my play demo](https://via.placeholder.com/800x400?text=Watch+Boxing+AI+Demo)](!https://drive.google.com/file/d/14716OGsd0Bl2DVU9waerE_U_6jVZnpA7/view?usp=sharing)



## Raissa Experiments (10 Configurations)
The notebook used is:
- `experiments/raissa_experiments.ipynb`

It now defines 10 unique experiment combinations including both policy types:
- 8 runs with `CnnPolicy`
- 2 runs with `MlpPolicy`

## Hyperparameter Results Table
After running the notebook, use:
- `results/raissa/tables/raissa_hyperparameter_results.csv`
- `results/raissa/tables/raissa_hyperparameter_results.md`

Paste the generated markdown table below for final submission:

| name | policy | learning_rate | gamma | batch_size | exploration_initial_eps | exploration_final_eps | exploration_fraction | mean_reward | std_reward | train_minutes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raissa-exp01-cnn-baseline | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp02-cnn-low-lr | CnnPolicy | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp03-cnn-higher-lr | CnnPolicy | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp04-cnn-gamma-097 | CnnPolicy | 1e-4 | 0.97 | 32 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp05-cnn-batch-64 | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp06-cnn-batch-128 | CnnPolicy | 1e-4 | 0.99 | 128 | 1.0 | 0.02 | 0.12 | - | - | - |
| raissa-exp07-cnn-slow-epsilon-decay | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.25 | - | - | - |
| raissa-exp08-cnn-fast-epsilon-decay | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.05 | - | - | - |
| raissa-exp09-mlp-baseline-ram | MlpPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | - | - | - |
| raissa-exp10-mlp-alt | MlpPolicy | 2e-4 | 0.98 | 128 | 1.0 | 0.02 | 0.15 | - | - | - |

## Noted Behavior / Insights
Fill after runs:
- Which hyperparameter changes improved performance?
- Which settings harmed performance?
- Why the final best configuration performed best?
- CNN vs MLP comparison result for Tennis.

## Required Artifacts
- Best model (assignment name): `results/raissa/models/dqn_model.zip`
- Best model copy: `results/raissa/models/best_dqn_tennis.zip`
- Hyperparameter table CSV: `results/raissa/tables/raissa_hyperparameter_results.csv`
- Hyperparameter table Markdown: `results/raissa/tables/raissa_hyperparameter_results.md`
- Gameplay video (optional from notebook cell): `results/raissa/videos/playback/*.mp4`

## Colab + Google Drive Export
The notebook includes export logic that copies key artifacts to:
- `/content/drive/MyDrive/Boxing_dqn_agent/raissa`

So you can access the model and results table even after Colab runtime disconnects.

