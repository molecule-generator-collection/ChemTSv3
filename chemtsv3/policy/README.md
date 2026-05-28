# chemtsv3.policy

Policy classes select child nodes during MCTS.

## Inheritance

```text
Policy
└── TemplatePolicy
    └── ScoreBasedPolicy
        └── UCT
            └── PUCT
                └── PUCTWithPredictor

UpperPredictor
└── LightGBMPredictor
```

## Policy (abstract)

Abstract base class for MCTS policies. Choose a child node by `select_child` method.

| Method | Description |
|---|---|
| `select_child(node: Node) -> Node` | Select one child of the given node. Must not be called if `node.children` is empty. |
| `observe(child: Node, objective_values: list[float], reward: float, is_filtered: bool)` | (Optional) Policies can update their internal state when observing the evaluation value of the node. By default, this method does nothing. |
| `analyze()` | (Optional) This method is called within MCTS.analyze(). Does nothing by default. |
| `candidates(node: Node) -> list[Node]` | (Optional) Return available child candidates. Returns all children by default. |

## TemplatePolicy (abstract)

Base policy with optional progressive widening.

| Parameter | Default | Description |
|---|---:|---|
| `pw_c` | `None` | Used for progressive widening. If set, the number of available child nodes is limited to `pw_c * (visit count ** pw_alpha) + pw_beta`. |
| `pw_alpha` | `None` | Progressive widening exponent. |
| `pw_beta` | `0` | Progressive widening offset. |
| `logger` | `None` | Logger used by the policy. Automatically set during YAML-based generation. |

## ScoreBasedPolicy (abstract)

Policy that selects the node with the highest value of `score` method. Supports epsilon-greedy selection.

| Parameter | Default | Description |
|---|---:|---|
| `pw_c` | `None` | Used for progressive widening. If set, the number of available child nodes is limited to `pw_c * (visit count ** pw_alpha) + pw_beta`. |
| `pw_alpha` | `None` | Progressive widening exponent. |
| `pw_beta` | `0` | Progressive widening offset. |
| `epsilon` | `0` | Probability of randomly selecting a child node while descending the search tree. |
| `logger` | `None` | Logger used by the policy. Automatically set during YAML-based generation. |

| Method | Description |
|---|---|
| `score(node: Node) -> float` | Return the selection score of the given child node. |

## UCT

Upper Confidence Bound applied to Trees policy.

| Parameter | Default | Description |
|---|---:|---|
| `c` | `0.3` | Weight of the exploration term. Higher values place more emphasis on exploration over exploitation. Can be a float, a depth-dependent callable, or a list of `(x, y)` points. |
| `best_rate` | `0.0` | Value between 0 and 1. The exploitation term is `best_rate * best reward + (1 - best_rate) * average reward`. |
| `max_prior` | `None` | Lower bound for the best reward. If the actual best reward is lower, this value is used instead. |
| `pw_c` | `None` | Used for progressive widening. If set, the number of available child nodes is limited to `pw_c * (visit count ** pw_alpha) + pw_beta`. |
| `pw_alpha` | `None` | Progressive widening exponent. |
| `pw_beta` | `0` | Progressive widening offset. |
| `epsilon` | `0` | Probability of randomly selecting a child node while descending the search tree. |
| `logger` | `None` | Logger used by the policy. Automatically set during YAML-based generation. |

## PUCT

Modified PUCT introduced in AlphaGo Zero.

`PUCT` inherits `UCT.__init__` without adding parameters.

| Parameter | Default | Description |
|---|---:|---|
| `c` | `0.3` | Weight of the exploration term. |
| `best_rate` | `0.0` | Value between 0 and 1 for mixing best reward and average reward in exploitation. |
| `max_prior` | `None` | Lower bound for the best reward. |
| `pw_c` | `None` | Used for progressive widening. If set, the number of available child nodes is limited to `pw_c * (visit count ** pw_alpha) + pw_beta`. |
| `pw_alpha` | `None` | Progressive widening exponent. |
| `pw_beta` | `0` | Progressive widening offset. |
| `epsilon` | `0` | Probability of random child selection. |
| `logger` | `None` | Logger used by the policy. Automatically set during YAML-based generation. |

## PUCTWithPredictor

PUCT variant that uses `{predicted evaluation value + exploration term}` as the score for nodes with zero visits instead of infinity. Currently supports subclasses of `MolStringNode`; `get_feature_vector()` can be overridden for other node classes. To make use of this policy, `MCTS.n_eval_width` must be set to `0`.

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `0.9` | Quantile level for the predictor, representing the target percentile of the response variable to estimate and use. Set to `0.5` when using a mean predictor so pinball loss can account for that. |
| `score_threshold` | `0.6` | If the recent prediction score, `1 - pinball loss / baseline pinball loss`, is better than this threshold, the model output is used by the policy. |
| `n_warmup_steps` | `None` | Number of observations before the first predictor training. If omitted, `batch_size` is used. |
| `batch_size` | `500` | Number of new observations required for subsequent predictor training. |
| `score_calculation_interval` | `25` | Number of prediction-target pairs between prediction score checks. |
| `score_calculation_window` | `200` | Number of recent prediction-target pairs used for score calculation. |
| `predictor_type` | `"lightgbm"` | Predictor implementation. Currently `"lightgbm"` is supported. |
| `predictor_params` | `None` | Parameters passed to the predictor. |
| `fp_radius` | `2` | Morgan fingerprint radius. Used only when `fp_size > 0` and the node is a subclass of `MolStringNode`. |
| `fp_size` | `2048` | Morgan fingerprint size. Set to `0` to disable fingerprints and use RDKit descriptors only. |
| `c` | `0.3` | Weight of the exploration term. |
| `best_rate` | `0.0` | Value between 0 and 1 for mixing best reward and average reward in exploitation. |
| `max_prior` | `None` | Lower bound for the best reward. |
| `pw_c` | `None` | Used for progressive widening. If set, the number of available child nodes is limited to `pw_c * (visit count ** pw_alpha) + pw_beta`. |
| `pw_alpha` | `None` | Progressive widening exponent. |
| `pw_beta` | `0` | Progressive widening offset. |
| `epsilon` | `0` | Probability of random child selection. |
| `logger` | `None` | Logger used by the policy. Automatically set during YAML-based generation. |
