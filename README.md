# ChemTSv3
Temporary repository for ChemTSv3

## Setup

<details>
  <summary><b>Minimal installation (Mac, Linux)</b></summary><br>

### Available classes
- **Transition**: `GBGATransition`, `GPT2Transition`, `RNNBasedMutation`, `RNNTransition`, `SMIRKSTransition`
- **Reward**: `GFPReward`, `SimilarityReward`, `JScoreReward`, `LogPReward`
- **Policy**: `UCT`, `PUCT`
- The corresponding Node classes and all implemented Filter classes are also available in this environment.

### Setup steps

1. Clone the repository
2. Install uv: https://docs.astral.sh/uv/getting-started/installation/
3. Restart the shell
4. Move to the repository root (e.g., cd molgen)
5. Run the following commands:
```bash
uv venv --python 3.11.11
source .venv/bin/activate
uv pip install pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
```

To activate the virtual environment, run the following command from the repository root (this process can also be automated through VS Code settings):
```bash
source .venv/bin/activate
```
To deactivate the virtual environment, run:
```bash
deactivate
```
</details>

<details>
  <summary><b>Minimal installation (Windows)</b></summary><br>

### Available classes
- **Transition**: `GBGATransition`, `GPT2Transition`, `RNNBasedMutation`, `RNNTransition`, `SMIRKSTransition`
- **Reward**: `GFPReward`, `SimilarityReward`, `JScoreReward`, `LogPReward`
- **Policy**: `UCT`, `PUCT`
- The corresponding Node classes and all implemented Filter classes are also available in this environment.

### Setup steps

1. Clone the repository
2. Install uv: https://docs.astral.sh/uv/getting-started/installation/
3. Restart the shell (and VSCode if used)
4. Move to the repository root (e.g., cd molgen)
5. Run the following commands:
```bash
uv venv --python 3.11.11
.venv\Scripts\activate
uv pip install pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
```

To activate the virtual environment, run the following command from the repository root (this process can also be automated through VS Code settings):
```bash
.venv\Scripts\activate
```
To deactivate the virtual environment, run:
```bash
deactivate
```
</details>

<details>
  <summary><b>Full installation (Mac, Linux)</b></summary><br>
  
### Available classes
- **Transition**: `BioT5Transition`, `ChatGPTTransition`, `ChatGPTTransitionWithMemory`, `GBGATransition`, `GPT2Transition`, `RNNBasedMutation`, `RNNTransition`, `SMIRKSTransition`
- **Reward**: `DScoreReward`, `DyRAMOReward`, `GFPReward`, `SimilarityReward`, `JScoreReward`, `LogPReward`, `TDCReward`
- The corresponding Node classes, along with all implemented Filter and Policy classes, are also available in this environment.
- `ChatGPTTransition` and `ChatGPTTransitionWithMemory` requires openai api key to use.

### Setup steps
1. Clone the repository
2. Install uv: https://docs.astral.sh/uv/getting-started/installation/
3. Restart the shell
4. Move to the repository root (e.g., cd molgen)
5. Run the following commands:
```bash
uv venv --python 3.11.11
source .venv/bin/activate
uv pip install pytdc==1.1.14 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 selfies==2.2.0 ipykernel==6.30.0 transformers==4.43.4 setuptools==78.1.1 lightgbm==4.6.0 openai==2.6.0 torch==2.5.1 --torch-backend=auto
```
To activate the virtual environment, run the following command from the repository root (this process can also be automated through VS Code settings):
```bash
source .venv/bin/activate
```
To deactivate the virtual environment, run:
```bash
deactivate
```
  
</details>
  
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
- `lightgbm` — required for **DScoreReward**, **DyRAMOReward**, **PUCTWithPredictor** / tested version: 3.3.5, 4.6.0
- `selfies` — required for **SELFIESStringNode** / tested version: 2.2.0
- `openai` — required for **ChatGPT2Transition**, **ChatGPT2TransitionWithMemory** / tested version: 2.6.0
- `pytdc` — required for **TDCReward** / tested version: 1.1.14
