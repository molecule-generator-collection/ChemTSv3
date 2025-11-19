## ChemTSv3
A unified tree search framework for molecular generation.
- **Node is modular**: Supports any molecular representation (e.g., SMILES, SELFIES, FASTA, or HELM) in either string or tensor format.
- **Transition is modular**: Allows any molecular transformation strategy, including graph-based editing, sequence generation with RNN or GPT-2, sequence mutation, or LLM-guided modification.
- **Filter is modular**: Enables flexible constraints such as structural alerts, scaffold preservation, or physicochemical property filters.
- **Reward is modular**: Anything can be optimized, including QSAR predictions or simulation results, for both single- and multi-objective tasks.

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
uv pip install numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
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
uv pip install numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
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
uv pip install pytdc==1.1.14 numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 selfies==2.2.0 ipykernel==6.30.0 transformers==4.43.4 setuptools==78.1.1 lightgbm==4.6.0 openai==2.6.0 torch==2.5.1 --torch-backend=auto
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
  <summary><b>Troubleshooting</b></summary><br>
  
### CUDA not available
In some cases (for example, when setting up environments on a control node), it may be necessary to reinstall torch with a different backend to enable CUDA support. However, since major implemented classes (including `RNNTransition`) are likely to run faster on the CPU, this is not strictly required. After reinstalling torch, you may also need to downgrade numpy to version 1.26.4 if it was upgraded during the process.
</details>
  
</details>

## Generation via CLI
See `config/mcts/example.yaml` for setting options.
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
- **Generation via notebook**: `sandbox/generation.ipynb`

## Options
See `config/mcts/example.yaml` for an example and advanced options. More examples (settings used in the paper) can be found in `config/mcts/egfr_de_novo` and `config/mcts/egfr_lead_opt`.

All options for each component (class) are defined as arguments in the `__init__()` method of the corresponding class.

**Node / Transition**:
|Node class|Transition class|Description|
|---|---|---|
|`MolSentenceNode`|`RNNTransition`|For de novo generation. Uses the specified RNN (GRU / LSTM) model.|
|`MolSentenceNode`|`GPT2Transition`|For de novo generation. Uses the specified Transformer (GPT-2) model.|
|`CanonicalSMILESStringNode`|`GBGATransition`|For lead optimization. Uses GB-GA mutation rules.|
|`CanonicalSMILESStringNode`|`SMIRKSTransition`|For lead optimization. Uses the specified SMIRKS rules (e.g. MMP-based ones).|
|`SMILESStringNode`|`ChatGPTTransition`|For lead optimization. Uses the specified prompt(s). Requires OpenAI API key.|

**Policy**:
- `UCT`: Does not use transition probabilities. Performed better with `RNNTransition` in our testing.
- `PUCT`: Incorporates transition probabilities. Performed better with `GBGATransition` in our testing.
- `PUCTWithPredictor`: Trains a predictor from the generation history and uses it when the prediction score exceeds a threshold. This option adds a few seconds of overhead per generation (depending on the number of child nodes per transition and the computational cost of each prediction), and is recommended only when reward functions are expensive. For non-molecular nodes, a function that returns a feature vector must be defined  (see `policy/puct_with_predictor.py` for details.)

**Basic options**
|Class|Option|Default|Description|
|---|---|---|---|
|-|`max_generations`|-|Stops generation after producing the specified number of molecules.|
|-|`time_limit`|-|Stops generation once the time limit (in seconds) is reached.|
|-|`root`|""|Key (string) for the root node (e.g. Canonical SMILES of the starting molecule for `CanonicalSMILESStringNode`). If `root` is not specified, an empty string "" will be used as the root node's key.|
|`MCTS`|`n_eval_width`|∞|By default (= ∞), evaluates all new leaf nodes after each transition. Setting `n_eval_width = 1` often improves sample efficiency and can be beneficial when reward computation is expensive.|
|`MCTS`|`filter_reward`|0|Substitutes the reward with this value when nodes are filtered. Use a list to specify different reward values for each filtering step. Set to "ignore" to skip reward assignment (in this case, other penalty types for filtered nodes, such as `failed_parent_reward`, needs to be set).|
|`UCT`, `PUCT`|`c`|0.3|A larger value prioritizes exploration over exploitation. Recommended range: 0.01–1.|
|`UCT`, `PUCT`|`best_rate`|0|A value between 0 and 1. The exploitation term is calculated as: `best_rate` * {best reward} + (1 - `best_rate`) * {average reward}. For better sample efficiency, it might be better to set this value to around 0.5 for de novo generations, and around 0.9 for lead optimizations.|

**Advanced options**
|Class|Option|Default|Description|
|---|---|---|---|
|`MCTS`|`n_eval_iters`|1|The number of child node evaluations. This value should not be >1 unless the evaluations are undeterministic (e.g. involve rollouts).|
|`MCTS`|`n_tries`|1|The number of attempts to obtain an unfiltered node in a single evaluation. This value should not be >1 unless the evaluations are undeterministic (e.g. involve rollouts).|
|`MCTS`|`allow_eval_overlaps`|False|Whether to allow overlap nodes when sampling eval candidates (recommended: False)|
|`MCTS`|`reward_cutoff`|None|Child nodes are removed if their reward is lower than this value. This applies only to nodes for which `has_reward() = True` (i.e., complete molecules). |
|`MCTS`|`reward_cutoff_warmups`|None|If specified, reward_cutoff will be inactive until `reward_cutoff_warmups` generations.|
|`MCTS`|`cut_failed_child`|False|If True, child nodes will be removed when {`n_eval_iters` * `n_tries`} evals are filtered.|
|`MCTS`|`failed_parent_reward`|"ignore"|Backpropagate this value when {`n_eval_width` * `n_eval_iters` * `n_tries`} evals are filtered from the node.|
|`MCTS`|`terminal_reward`|"ignore"|If a float value is set, that value is backpropagated when a leaf node reaches a terminal state. If set to "ignore", no value is backpropagated.|
|`MCTS`|`cut_terminal`|True|If True, terminal nodes are pruned from the search tree and will not be visited more than once.|
|`MCTS`|`avoid_duplicates`|True|If True, duplicate nodes won't be added to the search tree. Should be True if the transition forms a cyclic graph. Unneeded if the tree structure of the transition graph is guranteed, and can be set to False to reduce memory usage.|
|`MCTS`|`discard_unneeded_states`|True|If True, discards node variables that are no longer needed after expansion. Set this to False when using custom classes that utilize these values.|
|`UCT`, `PUCT`|`pw_c`, `pw_alpha`, `pw_beta`|None, 0, 0|If `pw_c` is set, the number of available child nodes is limited to `pw_c` * ({visit count} ** `pw_alpha`) + `pw_beta`.|
|`UCT`, `PUCT`|`max_prior`|None (0)|A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.|
|`UCT`, `PUCT`|`epsilon`|0|The probability of randomly selecting a child node while descending the search tree.|
|`RNNTransition`, `GPT2Transition`|`top_p`|0.995| Nucleus sampling threshold in (0, 1]; keeps the smallest probability mass ≥ `top_p`.|
|`RNNTransition`, `GPT2Transition`|`temperature`|1| Logit temperature > 0 applied **before** top_p; values < 1.0 sharp, > 1.0 smooth.|
|`RNNTransition`|`sharpness`|1| Probability distribution sharpness > 0 applied **after** top_p; values < 1.0 smooth, > 1.0 sharp.|
|`RNNTransition`|`disable_top_p_on_rollout`|False| If True, top_p won't be applied for rollouts.|
|`SMIRKSTransition`|`limit`|None| If the number of generated SMILES exceeded this value, stops applying further SMIRKS patterns. The order of SMIRKS patterns are shuffled with weights before applying transition if this option is enabled.|

## Model training
- **RNN (GRU) training** (example): `python sandbox/model_training.py -c config/training/train_rnn_smiles.yaml`
- **Transformer (GPT-2) training** (example): `python sandbox/model_training.py -c config/training/train_gpt2.yaml`
Change `dataset_path` in YAML to train on an arbitrary dataset (1 sentence per line).

## Optional dependencies
|Package|Required for|Tested version|
|---|---|---|
|`lightgbm`|`DScoreReward`, `DyRAMOReward`, `PUCTWithPredictor`|3.3.5, 4.6.0|
|`selfies`|`SELFIESStringNode`|2.2.0|
|`openai`|`ChatGPT2Transition`, `ChatGPT2TransitionWithMemory`|2.6.0|
|`pytdc`|`TDCReward`|1.1.14|

