# ChemTSv3
Temporary repository for ChemTSv3

## Environments

<details>
  <summary>Setting up a minimal conda-forge environment</summary>

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
  <summary>Setting up a conda-forge environment for benchmark replication</summary>

This section explains how to set up a conda-forge environment to replicate benchmark results.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `JensenTransition`(, `SMIRKSTransition`)
- **Reward**: `GuacaMolReward`, `TDCReward`(, `JScoreReward`, `LogPReward`) 
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
