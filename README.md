# DQN Atari Assignment - Tennis 

## Environment
- Gymnasium Atari environment: `ALE/Tennis-v5`
- Framework: Stable-Baselines3 `DQN`

## Scripts
- `scripts/train.py`: trains DQN and saves model artifacts
- `scripts/play.py`: loads trained model and runs evaluation gameplay

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
- `/content/drive/MyDrive/Tenis_dqn_agent/raissa`

So you can access the model and results table even after Colab runtime disconnects.

## Suggested Run Order (Notebook)
1. Cell 1: Setup + Drive mount
2. Cell 2: Utilities
3. Cell 3: Experiment definitions
4. Cell 4: Training runs (10 configs)
5. Cell 5: Save result tables
6. Cell 6: Visual comparison
7. Cell 7: Export artifacts and optional gameplay recording
