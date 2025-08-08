# ChemTSv3
Temporary repository for ChemTSv3

## Generation via CLI
See `config/mcts/example.yaml` and `config/mcts/example_chain_*.yaml` for setting options.
```bash
# Simple generation
python sandbox/generation.py -c config/mcts/example.yaml
# Chain generation
python sandbox/generation.py -c config/mcts/chain_example_1.yaml
# Load
python sandbox/generation.py -l sandbox/~~~/save --max_generations 100 --time_limit 60
```

## Notebooks
All notebooks are located in the `sandbox` directory.
- **Tutorial**: `tutorial_user_1.ipynb`, `tutorial_user_2.ipynb` and `tutorial_dev.ipynb`
- **YAML-based Generation**: `generation.ipynb`
- **Model training**: `train_rnn.ipynb` and `train_gpt2.ipynb`

## Environments

<details>
  <summary><b>Setting up a minimal conda-forge environment</b></summary><br>

This section explains how to set up a minimal conda-forge environment. This environment can run all tutorial notebooks.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `JensenTransition`, `SMIRKSTransition`
- **Reward**: `JScoreReward`, `LogPReward`
- The corresponding Node classes, along with all implemented Filter and Policy classes, are also available in this environment.

### Setup steps

```bash
conda create -n v3env-m python=3.11.13
conda activate v3env-m
conda install -c conda-forge ipykernel rdkit transformers pytorch pytorch-gpu
```
Note: For CPU-only environments, omit pytorch-gpu from the last command.
</details>

<details>
  <summary><b>Setting up a conda-forge environment for benchmark replication</b></summary><br>

This section explains how to set up a conda-forge environment to replicate benchmark results.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `JensenTransition`, `SMIRKSTransition`
- **Reward**: `GuacaMolReward`, `TDCReward`, `JScoreReward`, `LogPReward`
- The corresponding Node classes, along with all implemented Filter and Policy classes, are also available in this environment.

### Setup steps

```bash
conda create -n v3env-b python=3.11.13
conda activate v3env-b
conda install -c conda-forge pytdc=1.1.14 guacamol ipykernel
conda install -c conda-forge pytorch pytorch-gpu
```
Note: For CPU-only environments, omit pytorch-gpu from the last command.
</details>
