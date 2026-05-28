# chemtsv3.transition

Transition classes define how child nodes are generated. 

## Inheritance

```text
Transition
├── AutoRegressiveTransition
│   ├── GPT2Transition
│   └── RNNTransition
├── ProtGPT2Transition
└── TemplateTransition
    ├── GBGATransition
    ├── GBGMTransition
    ├── RNNBasedMutation
    ├──SMIRKSTransition
    └── BlackBoxTransition
       └── LLMTransition
           ├── BioT5Transition
           ├── ChatGPTTransition
           └── ChatGPTTransitionWithMemory
```

## Transition (abstract)

Abstract base class for transitions.

| Parameter | Default | Description |
|---|---:|---|
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

| Method | Description |
|---|---|
| `next_nodes(node: Node)` | Return the list of the child nodes. If the node is terminal, an empty list `[]` should be returned. |
| `rollout(initial_node: Node)` | (Optional) Sample an offspring node that satisfies `has_reward() = True`. By default, this method repeatedly calls `next_nodes()`. |
| `observe(node: Node, objective_values: list[float], reward: float, is_filtered: bool)` | (Optional) Transitions can update their internal state when observing the reward of the node. By default, this method does nothing. |
| `analyze()` | (Optional) This method is called within Generation.analyze(). Does nothing by default. |

## AutoRegressiveTransition (abstract)

Base class for autoregressive transitions.

| Parameter | Default | Description |
|---|---:|---|
| `lang` | required | Language object used to convert between tokens, tensors, and sentences. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## RNNTransition

Autoregressive transition backed by `RNNLanguageModel`.

| Parameter | Default | Description |
|---|---:|---|
| `lang` | required | Inherited from `AutoRegressiveTransition`. Language object used by the model. |
| `model` | `None` | `RNNLanguageModel` instance. Specify one or none of `model` and `model_dir`, not both. |
| `model_dir` | `None` | Directory containing `model.pt` and `config.json`. Specify one or none of `model` and `model_dir`, not both. |
| `device` | `None` | Torch device specification, e.g. `"cpu"`, `"cuda"`, or `"cuda:0"`. |
| `max_length` | `None` | Maximum rollout length. If omitted, an effectively unlimited value is used. |
| `top_p` | `0.995` | Nucleus sampling threshold in `(0, 1]`; keeps the smallest probability mass greater than or equal to `top_p`. Set to `1.0` to disable. |
| `temperature` | `1.0` | Logit temperature greater than 0 applied before top-p; values less than 1.0 sharpen, values greater than 1.0 smooth. |
| `sharpness` | `1.0` | Probability distribution sharpness greater than 0 applied after top-p; values less than 1.0 smooth, values greater than 1.0 sharpen. |
| `disable_top_p_on_rollout` | `False` | If True, top-p is not applied during rollouts. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## GPT2Transition

Autoregressive transition backed by a GPT-2 language model.

| Parameter | Default | Description |
|---|---:|---|
| `lang` | required | Inherited from `AutoRegressiveTransition`. Language object used by the model. |
| `model` | `None` | GPT-2 model instance. Specify either `model` or `model_dir`, not both. |
| `model_dir` | `None` | Directory from which a GPT-2 model is loaded. Specify either `model` or `model_dir`, not both. |
| `device` | `None` | Torch device specification, e.g. `"cpu"`, `"cuda"`, or `"cuda:0"`. |
| `logger` | `None` | Inherited from `Transition`. |
| `temperature` | `1.0` | Logit temperature greater than 0 applied before top-p; values less than 1.0 sharpen, values greater than 1.0 smooth. |
| `top_p` | `0.995` | Nucleus sampling threshold in `(0, 1]`; keeps the smallest probability mass greater than or equal to `top_p`. Set to `1.0` to disable. |
| `top_k` | `0` | Rollout generation top-k setting. Inactive when set to 0. |
| `repetition_penalty` | `1.0` | Rollout generation repetition penalty. Inactive when set to 1.0. |

## ProtGPT2Transition

FASTA transition backed by the `nferruz/ProtGPT2` text-generation model.

| Parameter | Default | Description |
|---|---:|---|
| `rollout_top_k` | `950` | Top-k setting used during ProtGPT2 rollout generation. |
| `max_length` | `100` | Maximum generated sequence length. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## TemplateTransition (abstract)

Base transition that applies transition-level filters, normalizes child probabilities, and optionally applies top-p pruning.

| Parameter | Default | Description |
|---|---:|---|
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `top_p` | `None` | Optional top-p threshold for pruning generated children by cumulative transition probability. Must be in `(0, 1]` when specified. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

| Method | Description |
|---|---|
| `_next_nodes_impl(node)` | Implement this method instead of `next_nodes()`. |

## GBGATransition

Graph-based genetic algorithm transition based on Jan H. Jensen's GB-GA implementation.

| Parameter | Default | Description |
|---|---:|---|
| `base_chances` | `[0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]` | Chances of `[insert_atom, change_bond_order, delete_cyclic_bond, add_ring, delete_atom, change_atom, append_atom]`. |
| `check_size` | `False` | Whether to use the molecule size filter. |
| `average_size` | `50.0` | Used for the molecule size filter only if `check_size` is True. |
| `size_std` | `5.0` | Used for the molecule size filter only if `check_size` is True. |
| `check_ring` | `True` | Whether to apply ring validity checks. |
| `merge_duplicates` | `True` | Whether to merge duplicated generated SMILES and aggregate probabilities. |
| `record_actions` | `False` | If True, used SMIRKS will be recorded as actions in child nodes. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `top_p` | `None` | Optional top-p threshold for pruning generated children by cumulative transition probability. Must be in `(0, 1]` when specified. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## GBGMTransition

Graph-based generative model transition based on Jan H. Jensen's GB-GM implementation.

| Parameter | Default | Description |
|---|---:|---|
| `size_mean` | `39.15` | Used to determine the target molecule size. |
| `size_std` | `3.50` | Used to determine the target molecule size. |
| `max_children` | `25` | Maximum number of child nodes generated during expansion. |
| `prob_ring_atom` | `0.63` | Probability of adding a ring atom. |
| `prob_double` | `0.8` | Probability mass assigned to double-bond ring reactions after rescaling. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `top_p` | `None` | Optional top-p threshold for pruning generated children by cumulative transition probability. Must be in `(0, 1]` when specified. |
| `max_expansion_tries` | `1000` | Maximum number of attempts to generate children during expansion. |
| `max_depth` | `200` | Maximum depth before terminating a molecule by appending EOS. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## SMIRKSTransition

Transition that applies weighted SMIRKS reactions to canonical SMILES nodes.

| Parameter | Default | Description |
|---|---:|---|
| `smirks_path` | `None` | Path to a `.txt` file containing SMIRKS patterns, one per line. Empty lines and text after `##` are ignored. Optional weights can be specified after `//`; default weight is `1.0`. Specify either `smirks_path` or `weighted_smirks`, not both. |
| `weighted_smirks` | `None` | List of `(SMIRKS, weight)` tuples. Can be provided instead of `smirks_path`. Specify either `smirks_path` or `weighted_smirks`, not both. |
| `limit` | `None` | If the number of generated SMILES exceeds this value, stops applying further SMIRKS patterns. When enabled, SMIRKS patterns are shuffled with weights before transition application. |
| `without_Hs` | `True` | If True, SMIRKS reactions are applied to the molecule without explicit hydrogens. |
| `with_Hs` | `False` | If True, SMIRKS reactions are applied to the molecule with explicit hydrogens via `Chem.AddHs`. |
| `kekulize` | `True` | Whether to kekulize the input molecule before applying reactions. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `top_p` | `None` | Optional top-p threshold for pruning generated children by cumulative transition probability. Must be in `(0, 1]` when specified. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |
| `record_actions` | `True` | If True, used SMIRKS patterns are recorded as actions in child nodes. |
| `output_dir` | `None` | Directory where SMIRKS statistics are saved. |

## RNNBasedMutation

Transition that uses an RNN to propose same-length sequences and converts differences into mutation candidates. The class docstring notes that the RNN should return a sentence of the same length with a reasonable chance.

| Parameter | Default | Description |
|---|---:|---|
| `n_samples` | `1` | Number of RNN rollout samples used to propose mutations. |
| `n_tries` | `5` | Maximum attempts to obtain an RNN rollout with the same length as the base sequence. |
| `model_dir` | `None` | Directory used to find the language file and load the RNN model. |
| `device` | `None` | Torch device specification passed to `RNNTransition` and `MolSentenceNode`. |
| `rnn_top_p` | `1.0` | Top-p value passed to the internal `RNNTransition`. |
| `rnn_temperature` | `1.0` | Temperature passed to the internal `RNNTransition`. |
| `rnn_sharpness` | `1.0` | Sharpness passed to the internal `RNNTransition`. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## BlackBoxTransition (abstract)

Template transition for black-box samplers that repeatedly sample child nodes.

| Parameter | Default | Description |
|---|---:|---|
| `n_samples` | `2` | Number of transition samples to request for each input node. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## LLMTransition (abstract)

Black-box transition driven by prompts. `###STRING###` in a prompt is replaced with the node string, and `###KEY###` is replaced with the node key.

| Parameter | Default | Description |
|---|---:|---|
| `prompt` | required | Prompt string or list of prompt strings used to request new candidates. |
| `n_samples` | `1` | Number of samples per prompt call. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## ChatGPTTransition

LLM transition backed by the OpenAI Responses API.

| Parameter | Default | Description |
|---|---:|---|
| `prompt` | required | Prompt string or list of prompt strings used to request new candidates. |
| `model` | `"gpt-4o-mini"` | OpenAI model name. |
| `api_key` | `None` | OpenAI API key. Specify either `api_key` or `api_key_path`. |
| `api_key_path` | `None` | Path to a text file containing the OpenAI API key. Specify either `api_key` or `api_key_path`. |
| `n_samples` | `1` | Number of samples per prompt call. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## ChatGPTTransitionWithMemory

ChatGPT transition that keeps conversation history and feeds back observed rewards. Class docstring: keeps conversation.

| Parameter | Default | Description |
|---|---:|---|
| `prompt` | required | Prompt string or list of prompt strings used to request new candidates. |
| `initial_prompt` | `None` | Optional first prompt used to initialize the conversation memory. |
| `model` | `"gpt-4o-mini"` | OpenAI model name. |
| `api_key` | `None` | OpenAI API key. Specify either `api_key` or `api_key_path`. |
| `api_key_path` | `None` | Path to a text file containing the OpenAI API key. Specify either `api_key` or `api_key_path`. |
| `n_samples` | `1` | Number of samples per prompt call. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |

## BioT5Transition

LLM transition backed by the `QizhiPei/biot5-base-text2mol` text-to-molecule model.

| Parameter | Default | Description |
|---|---:|---|
| `prompt` | required | Prompt string or list of prompt strings used to request new candidates. |
| `n_samples` | `1` | Inherited from `BlackBoxTransition`. |
| `filters` | `None` | Filters applied to generated child nodes inside the transition. |
| `logger` | `None` | Logger used by the transition. Automatically set during YAML-based generation. |
