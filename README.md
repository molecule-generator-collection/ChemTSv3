# ChemTSv3
Temporary repository for ChemTSv3

## Environments

<details>
  <summary>Setting up a conda-forge environment for benchmark replication</summary>

This section explains how to set up a `conda` environment to replicate benchmark results.

### Available classes
- **Transition**: `RNNTransition`, `GPT2Transition`, `JensenTransition`(, `SMIRKSTransition`)
- **Reward**: `GuacaMolReward`, `TDCReward`(, `JScoreReward`, `LogPReward`) 
- All implemented classes of `Node`, `Filter`, and `Policy` are also available.

### Setup steps

```bash
conda create -n v3env python=3.11.13
conda activate v3env
conda install -c conda-forge pytdc=1.1.14 guacamol ipykernel
conda install -c conda-forge pytorch pytorch-gpu
```
Note: For CPU-only environments, omit pytorch-gpu from the last command.
</details>
