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
uv pip install chemtsv3 numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
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
uv pip install chemtsv3 numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 ipykernel==6.30.0 transformers==4.43.4 torch==2.5.1 --torch-backend=auto
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
uv pip install chemtsv3 pytdc==1.1.14 numpy==1.26.4 pandas==2.3.3 matplotlib==3.10.7 rdkit==2023.09.6 selfies==2.2.0 ipykernel==6.30.0 transformers==4.43.4 setuptools==78.1.1 lightgbm==4.6.0 openai==2.6.0 torch==2.5.1 --torch-backend=auto
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
  <summary><b>Optional dependencies</b></summary><br>

The full installation includes the following optional packages:

|Package|Required for|Tested version|
|---|---|---|
|`lightgbm`|`DScoreReward`, `DyRAMOReward`, `PUCTWithPredictor`|3.3.5, 4.6.0|
|`selfies`|`SELFIESStringNode`|2.2.0|
|`openai`|`ChatGPT2Transition`, `ChatGPT2TransitionWithMemory`|2.6.0|
|`pytdc`|`TDCReward`|1.1.14|

</details>

<details>
  <summary><b>Troubleshooting</b></summary><br>
  
### CUDA not available
In some cases (for example, when setting up environments on a control node), it may be necessary to reinstall torch with a different backend to enable CUDA support. However, since major implemented classes (including `RNNTransition`) are likely to run faster on the CPU, this is not strictly required. After reinstalling torch, you may also need to downgrade numpy to version 1.26.4 if it was upgraded during the process.
</details>
  
</details>

## Generation via CLI
See `config/mcts/example.yaml` for an example YAML configuration.
```bash
# Simple generation
chemtsv3 -c config/mcts/example.yaml
# Chain generation
chemtsv3 -c config/mcts/example_chain_1.yaml
# Load a checkpoint and continue the generation
chemtsv3 -l generation_results/~~~ --max_generations 100 --time_limit 60
```

## Notebooks
- **Tutorials**: `sandbox/tutorial/***.ipynb`
- **Generation via notebook**: `sandbox/generation.ipynb`

## Options
See `config/mcts/example.yaml` for an example and advanced options. More examples (settings used in the paper) can be found in `config/mcts/egfr_de_novo/` and `config/mcts/egfr_lead_opt/`.

All options for each component (class) are defined as arguments in the `__init__()` method of the corresponding class.

<details>
  <summary><b>Nodes and Transitions</b></summary><br>

**For general usage:**
|Node class|Transition class|Description|
|---|---|---|
|`MolSentenceNode`|`RNNTransition`|For de novo generation. Uses the RNN (GRU / LSTM) model specified by `model_dir`.|
|`MolSentenceNode`|`GPT2Transition`|For de novo generation. Uses the Transformer (GPT-2) model specified by `model_dir`.|
|`CanonicalSMILESStringNode`|`GBGATransition`|For lead optimization. Uses [GB-GA mutation rules](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c).|
|`CanonicalSMILESStringNode`|`SMIRKSTransition`|For lead optimization. Uses the specified SMIRKS rules (e.g. MMP-based ones).|
|`SMILESStringNode`|`ChatGPTTransition`|For lead optimization. Uses the specified prompt(s) as input to the GPT model specified by `model` (e.g., `"gpt-4o-mini"`). Requires an OpenAI API key.|

**For research purposes (did not perform well in our testing):**
|Node class|Transition class|Description|
|---|---|---|
|`CanonicalSMILESStringNode`|`GBGMTransition`|For de novo generation. Uses [GB-GM rules](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c). Rollouts iteratively apply transitions until the molecule size reaches a sampled value determined by `size_mean` and `size_std`.|
|`FASTAStringNode`|`ProtGPT2Transition`|For de novo protein generation. Uses the [ProtGPT2 model](https://www.nature.com/articles/s41467-022-32007-7).|
|`SELFIESStringNode`|`BioT5Transition`|For lead optimization. Uses the specified prompt(s) as input to the [BioT5 text2mol model](https://github.com/QizhiPei/BioT5).|
|`SMILESStringNode`|`ChatGPTTransitionWithMemory`|For lead optimization. Unlike `ChatGPTTransition`, retains conversation history and feedback reward calculation results to the model.|

</details>

<details>
  <summary><b>Policies</b></summary><br>

- `UCT`: Does not use transition probabilities. Performed better with `RNNTransition` in our testing.
- `PUCT`: Incorporates transition probabilities (follows the modification introduced in [AlphaGo Zero](https://www.nature.com/articles/nature24270)). Performed better with `GBGATransition` in our testing.
- `PUCTWithPredictor`: Trains an optimistic predictor of leaf-node evaluations using the generation history, and uses its output as the score for unvisited nodes when the model’s performance (measured by the normalized pinball loss) exceeds a specified threshold. This option adds a few seconds of overhead per generation (depending on the number of child nodes per transition and the computational cost of each prediction), and is recommended only when the reward calculations are expensive. Inherits all the arguments of `UCT` and `PUCT`. For non-molecular nodes, a function that returns a feature vector must be defined  (see `policy/puct_with_predictor.py` for details.)

</details>

<details>
  <summary><b>Basic options</b></summary><br>

|Class|Option|Default|Description|
|---|---|---|---|
|-|`max_generations`|-|Stops generation after producing the specified number of molecules.|
|-|`time_limit`|-|Stops generation once the time limit (in seconds) is reached.|
|-|`root`|`""`|Key (string) for the root node (e.g. SMILES of the starting molecule for `SMILESStringNode`). Multiple roots can be specified by list input. If not specified, an empty string `""` will be used as the root node's key.|
|`MCTS`|`n_eval_width`|∞|By default (= ∞), evaluates all new leaf nodes after each transition. Setting `n_eval_width = 1` often improves sample efficiency and can be beneficial when reward computation is expensive.|
|`MCTS`|`filter_reward`|0|Substitutes the reward with this value when nodes are filtered. Use a list to specify different reward values for each filtering step. Set to `"ignore"` to skip reward assignment (in this case, other penalty types for filtered nodes, such as `failed_parent_reward`, needs to be set).|
|`UCT`, `PUCT`, `PUCTWithPredictor`|`c`|0.3|A larger value prioritizes exploration over exploitation. Recommended range: [0.01, 1]|
|`UCT`, `PUCT`, `PUCTWithPredictor`|`best_rate`|0|A value between 0 and 1. The exploitation term is calculated as: `best_rate` * {best reward} + (1 - `best_rate`) * {average reward}. For better sample efficiency, it might be better to set this value to around 0.5 for de novo generations, and around 0.9 for lead optimizations.|

</details>

<details>
  <summary><b>Advanced options</b></summary><br>


For other options and further details, please refer to each class’s `__init__()` method.


|Class|Option|Default|Description|
|---|---|---|---|
|-|`seed`|-|The seed value for `random`, `np.random` and `torch`.|
|-|`device`|-|Torch device specification (e.g., "cpu", "cuda", "cuda:0"). For `RNNTransition`, using the CPU tends to be faster.|
|-|`debug`|False|If True, debug logging are enabled.|
|-|`silent`|False|If True, console logging are disabled.|
|-|`next_yaml_path`|False|If a path to the YAML config for the next generator is set, the generated molecules will be passed for chain generation.|
|-|`n_keys_to_pass`|3|Number of top-k generated molecules (keys) to be used as root nodes for the next generator.|
|`MCTS`|`save​_on​_completion`|False|If True, saves a checkpoint upon completion of the generation.|
|`MCTS`|`n_eval_iters`|1|The number of child node evaluations. This value should not be > 1 unless the evaluations are undeterministic (e.g. involve rollouts).|
|`MCTS`|`n_tries`|1|The number of attempts to obtain an unfiltered node in a single evaluation. This value should not be >1 unless the evaluations are undeterministic (e.g. involve rollouts).|
|`MCTS`|`allow​_eval​_overlaps`|False|Whether to allow overlap nodes when sampling eval candidates (recommended: False)|
|`MCTS`|`reward_cutoff`|None|Child nodes are removed if their reward is lower than this value. This applies only to nodes for which `has_reward() = True` (i.e., complete molecules). |
|`MCTS`|`reward​_cutoff​_warmups`|None|If specified, reward_cutoff will be inactive until `reward_cutoff_warmups` generations.|
|`MCTS`|`cut_failed_child`|False|If True, child nodes will be removed when {`n_eval_iters` * `n_tries`} evals are filtered.|
|`MCTS`|`failed​_parent​_reward`|`"ignore"`|Backpropagate this value when {`n_eval_width` * `n_eval_iters` * `n_tries`} evals are filtered from the node.|
|`MCTS`|`terminal_reward`|`"ignore"`|If a float value is set, that value is backpropagated when a leaf node reaches a terminal state. If set to `"ignore"`, no value is backpropagated.|
|`MCTS`|`cut_terminal`|True|If True, terminal nodes are pruned from the search tree and will not be visited more than once.|
|`MCTS`|`avoid_duplicates`|True|If True, duplicate nodes won't be added to the search tree. Should be True if the transition forms a cyclic graph. Unneeded if the tree structure of the transition graph is guranteed, and can be set to False to reduce memory usage.|
|`MCTS`|`discard​_unneeded​_states`|True|If True, discards node variables that are no longer needed after expansion. Set this to False when using custom classes that utilize these values.|
|`UCT`, `PUCT`, `PUCTWithPredictor`|`pw_c`, `pw_alpha`, `pw_beta`|None, 0, 0|If `pw_c` is set, the number of available child nodes is limited to `pw_c` * ({visit count} ** `pw_alpha`) + `pw_beta`.|
|`UCT`, `PUCT`, `PUCTWithPredictor`|`max_prior`|None (0)|A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.|
|`UCT`, `PUCT`, `PUCTWithPredictor`|`epsilon`|0|The probability of randomly selecting a child node while descending the search tree.|
|`PUCTWithPredictor`|`alpha`|0.9|Quantile level for the predictor, representing the target percentile of the response variable to be estimated and used.|
|`PUCTWithPredictor`|`score_threshold`|0.6|If the recent prediction score (1 - {pinball loss} / {baseline pinball loss}) is better than this threshold, the model will be used afterwards.|
|`MolSentenceNode​`, `MolStringNode`|`use​_canonical​_smiles​_as​_key`|False|Whether to convert generated molecules to canonical SMILES when generating keys. If False, the same molecule may be counted multiple times.|
|`RNNTransition`, `GPT2Transition`|`top_p`|0.995|Nucleus sampling threshold in (0, 1]; keeps the smallest probability mass ≥ `top_p`.|
|`RNNTransition`, `GPT2Transition`|`temperature`|1|Logit temperature > 0 applied **before** `top_p`; values < 1.0 sharp, > 1.0 smooth.|
|`RNNTransition`|`sharpness`|1| Probability distribution sharpness > 0 applied **after** `top_p`; values < 1.0 smooth, > 1.0 sharp.|
|`RNNTransition`|`disable​_top​_p​_on​_rollout`|False|If True, `top_p` won't be applied for rollouts.|
|`SMIRKSTransition`|`limit`|None|If the number of generated SMILES exceeded this value, stops applying further SMIRKS patterns. The order of SMIRKS patterns are shuffled with weights before applying transition if this option is enabled.|

</details>

<details>
  <summary><b>Filters</b></summary><br>

**Sanity**
- `ValidityFilter`: Excludes invalid molecule objects. Since other filters and rewards typically assume validity and do not recheck it, usually this filter should be applied first in molecular generation.
- `RadicalFilter`: Excludes molecules whose number of radical electrons is not 0.
- `ConnectivityFilter`: Excludes molecules whose number of disconnected fragments is not 1.

**Topological**
- `SubstructureFilter`: Excludes molecules that **do not** contain the specified (list of) substructure(s) by `smiles` or `smarts` arguments. If `preserve` is set to False, excludes molecules that **do** contain the specified (list of) substructure(s) instead. By specifying appropriate SMARTS patterns, it is possible to control where substitutions or structural modifications (i.e., adding a substituent or arm) are allowed to occur.
- `AromaticRingFilter`: Excludes molecules whose number of aromatic rings falls outside the range [`min`, `max`]. (Default: [1, ∞))
- `HeavyAtomCountFilter`: Excludes molecules whose number of heavy atoms falls outside the range [`min`, `max`]. (Default: [0, 45])
- `MaxRingSizeFilter`: Excludes molecules whose largest ring size falls outside the range [`min`, `max`]. (Default: [0, 6])
- `MinRingSizeFilter`: Excludes molecules whose smallest ring size falls outside the range [`min`, `max`]. (Default: (-∞, ∞))
- `RingBondFilter`: Excludes molecules containing ring allenes (`[R]=[R]=[R]`) or double bonds in small rings (`[r3,r4]=[r3,r4]`).
- `RotatableBondsFilter`: Excludes molecules whose number of rotatable bonds falls outside the range [`min`, `max`]. (Default: [0, 10])

**Structural alert**
- `ROCFilter`: Excludes molecules that contain structural alerts defined by Ohta and Cho.
- `CatalogFilter`: Excludes molecules that contain structural alerts in the specified list of [rdkit.Chem.FilterCatalogParams.FilterCatalogs](https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html#rdkit.Chem.rdfiltercatalog.FilterCatalogParams.FilterCatalogs). (e.g. `catalogs = ["PAINS_A", "PAINS_B", "PAINS_C", "NIH", "BRENK"]`)

**Drug-likeness**
- `PubChemFilter`: Excludes molecules based on the frequency of occurrence of molecular patterns in the PubChem database. Reported in [Ma et al.](https://doi.org/10.1021/acs.jcim.1c00679).
- `LipinskiFilter`: Excludes molecules based on Lipinski’s Rule of Five. Set `rule_of` to 3 to apply the Rule of Three instead.
- `SAScoreFilter`: Excludes molecules whose synthetic accessibility score (SA Score) falls outside the range [`min`, `max`]. (Default: [1, 3.5])

**Physicochemical**
- `ChargeFilter`: Excludes molecules whose formal charge is not 0.
- `HBAFilter`: Excludes molecules whose number of hydrogen bond acceptors falls outside the range [`min`, `max`]. (Default: [0, 10])
- `HBDFilter`: Excludes molecules whose number of hydrogen bond donors falls outside the range [`min`, `max`]. (Default: [0, 5])
- `LogPFilter`: Excludes molecules whose LogP value falls outside the range [`min`, `max`]. (Default: (-∞, 5])
- `TPSAFilter`: Excludes molecules whose topological polar surface area (TPSA) falls outside the range [`min`, `max`]. (Default: [0, 140])
- `WeightFilter`: Excludes molecules whose molecular weight falls outside the range [`min`, `max`]. (Default: [0, 500])

**Misc**
- `KnownListFilter`: Excludes molecules that are contained in the key column of the input CSV file(s), and overrides their reward with the corresponding value from the reward column (unless applied for the transition). (CSV files from generation results can be used directly.)

Filters can also be specified using `filters` argument of transitions that inherit from `TemplateTransition` (e.g. `GBGATransition`, `SMIRKSTransition`, `ChatGPTTransition`) to directly exclude molecules from child nodes.

</details>

## Model training
- **RNN (GRU) training** (example): `chemtsv3-train -c config/training/train_rnn_smiles.yaml`
- **Transformer (GPT-2) training** (example): `chemtsv3-train -c config/training/train_gpt2.yaml`
Change `dataset_path` in YAML to train on an arbitrary dataset (1 sentence per line).
