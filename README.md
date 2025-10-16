# ChemTSv3
Temporary repository for ChemTSv3

## Setup

<details>
  <summary><b>Setting up a minimal conda-forge environment</b></summary><br>

This section explains how to set up a minimal conda-forge environment. This environment can run all tutorial notebooks.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `GBGATransition`, `SMIRKSTransition`
- **Reward**: `JScoreReward`, `LogPReward`
- The corresponding Node classes, along with all implemented Filter and Policy classes, are also available in this environment.

### Setup steps

```bash
conda create -n v3env-m python=3.11.13
conda activate v3env-m
conda install -c conda-forge ipykernel rdkit transformers pytorch
```
</details>

<details>
  <summary><b>Setting up a conda-forge environment for benchmark replication</b></summary><br>

This section explains how to set up a conda-forge environment to replicate benchmark results.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `GBGATransition`, `SMIRKSTransition`
- **Reward**: `TDCReward`, `JScoreReward`, `LogPReward`
- The corresponding Node classes, along with all implemented Filter and Policy classes, are also available in this environment.

### Setup steps

```bash
conda create -n v3env-b python=3.11.13
conda activate v3env-b
conda install -c conda-forge pytdc=1.1.14 ipykernel
```
Note: For CPU-only environments, omit pytorch-gpu from the last command.
</details>

## Generation via CLI
See `config/mcts/example.yaml` for setting options. For chain settings, refer to `config/mcts/example_chain_*.yaml`.
```bash
# Simple generation
python sandbox/generation.py -c config/mcts/example.yaml
# Chain generation
python sandbox/generation.py -c config/mcts/example_chain_1.yaml
# Load a checkpoint and continue the generation
python sandbox/generation.py -l sandbox/generation_result/~~~/checkpoint --max_generations 100 --time_limit 60
```

## Notebooks
- **Tutorials**: `sandbox/tutorial/***.ipynb`
- **YAML-based Generation**: `sandbox/generation.ipynb`
- **Model training**: `sandbox/train_rnn.ipynb` and `sandbox/train_gpt2.ipynb`

## Optional Dependencies
- `lightgbm==3.2.1~3.3.5` — required for **DScoreReward**, **DyRAMOReward**
- `selfies` — required for **SELFIESStringNode**  
- `openai` — required for **ChatGPT2Transition**  
- `tdc` — required for **TDCReward**
