# chemtsv3.generator

Generator classes control the generation / optimization loop.

## Inheritance

```text
Generator
├── MCTS
│   └── AsyncParallelMCTS
│       └── DoubleAsyncParallelMCTS
├── HeapQueueGenerator
└── RandomGenerator
```

## Generator (abstract)

Base generator class. Override `_generate_impl` and, when necessary, `__init__` to implement a generator.

| Parameter | Default | Description |
|---|---:|---|
| `transition` | required | Transition used to generate child nodes and rollouts. |
| `reward` | `LogPReward()` | Reward instance used to evaluate nodes. |
| `filters` | `None` | Filters applied before reward calculation. |
| `filter_reward` | `0` | Substitute reward value used when nodes are filtered. Set to `"ignore"` to skip reward assignment. Use a list to specify different rewards for each filter step. |
| `precalculated_csv_paths` | `None` | Paths of result CSV files of the previous runs with the same reward can be specified here so that their reward and objective values are reused instead of recalculated. Later files take priority when keys overlap. |
| `name` | `None` | Generator name. If omitted, a timestamp-based name is generated. |
| `output_dir` | `None` | Directory where generation results and logs are saved. |
| `logger` | `None` | Logger instance used to record generation results. |
| `logging_interval` | `None` | Number of generations between each log flush. Overrides `info_interval`. |
| `info_interval` | `100` | Number of generations between each logging of the generation result. |
| `analyze_interval` | `10000` | Number of generations between each call of `analyze()`. |
| `verbose_interval` | `None` | Number of generations between each verbose log output. |
| `save_interval` | `None` | Number of generations between each checkpoint save. |
| `save_on_completion` | `False` | If True, saves a checkpoint when generation completes. |
| `include_transition_to_save` | `False` | If True, includes the transition object in the checkpoint file. |

## MCTS

Performs Monte Carlo tree search to maximize the reward.

| Parameter | Default | Description |
|---|---:|---|
| `root` | required | Root node. In non-YAML generation, use `SurrogateNode` to search from multiple root nodes. In YAML-based generation, specify a (list of) node key(s) instead. |
| `transition` | required | Transition used to expand nodes and perform rollouts. |
| `reward` | `LogPReward()` | Reward instance used to evaluate nodes. |
| `policy` | `UCT()` | Policy used for child selection. |
| `filters` | `None` | Filters applied before reward calculation. |
| `filter_reward` | `0` | Substitutes the reward with this value when nodes are filtered. Use a list to specify different reward values for each filtering step. Set to `"ignore"` to skip reward assignment; in that case, other penalties such as `failed_parent_reward` may be needed. |
| `failed_parent_reward` | `"ignore"` | Backpropagates this value when all `{n_eval_width * n_eval_iters * n_tries}` evaluations from a node are filtered. Set to `-1` for ChemTSv2 replication. Unused for batch reward calculation. |
| `n_eval_width` | `float("inf")` | Number of children sampled during evaluation. Set to `0` to use the policy instead of sampling. Set to `float("inf")` in Python or `.inf` in YAML to evaluate all children. |
| `allow_eval_overlaps` | `False` | Whether to allow overlapping nodes when sampling evaluation candidates. Recommended: False. |
| `n_eval_iters` | `1` | Number of child node evaluations. Rollouts are used for children whose `has_reward()` is False. |
| `n_tries` | `1` | Number of attempts to obtain an unfiltered node in a single evaluation. This should usually be 1 unless `has_reward()` can be False or filters are probabilistic. |
| `cut_failed_child` | `False` | If True, child nodes are removed when `{n_eval_iters * n_tries}` evaluations are filtered. |
| `reward_cutoff` | `None` | Child nodes are removed if their reward is lower than this value. |
| `reward_cutoff_warmups` | `None` | If specified, `reward_cutoff` is inactive until this many generations have completed. |
| `terminal_reward` | `"ignore"` | If a float is set, that value is backpropagated when a leaf reaches a terminal state. If `"ignore"`, no value is backpropagated. Set to `-1` for ChemTSv2 replication. |
| `cut_terminal` | `True` | If True, terminal nodes are pruned from the search tree and will not be visited more than once. Set to False for ChemTSv2 replication. |
| `avoid_duplicates` | `True` | If True, duplicate nodes are not added to the search tree. Should be True when the transition forms a cyclic graph. Can be False to reduce memory when the transition graph is guaranteed to be a tree. |
| `discard_unneeded_states` | `None` | If True, discards variables of nodes that will no longer be used after expansion. Unused for batch reward calculation. Caches are handled independently. |
| `max_tree_depth` | `None` | Maximum tree depth to expand. |
| `virtual_loss` | `0.0` | For `BatchReward` or rewards whose `n_batch() > 1`, this value is temporarily used until enough nodes are pooled for reward calculation. |
| `use_dummy_reward` | `False` | If True, backpropagated value is fixed to 0 while rewards and objective values are still calculated. |
| `precalculated_csv_paths` | `None` | Paths of result CSV files of the previous runs with the same reward can be specified here so that their reward and objective values are reused instead of recalculated. Later files take priority when keys overlap. |
| `name` | `None` | Generator name. If omitted, a timestamp-based name is generated. |
| `output_dir` | `None` | Directory where generation results and logs are saved. |
| `logger` | `None` | Logger instance used to record generation results. Automatically set during YAML-based generation. |
| `logging_interval` | `None` | Number of generations between each log flush. Overrides `info_interval`. |
| `info_interval` | `100` | Number of generations between each logging of the generation result. |
| `analyze_interval` | `10000` | Number of generations between each call of `analyze()`. |
| `verbose_interval` | `None` | Number of generations between each verbose log output. |
| `save_interval` | `None` | Number of generations between each checkpoint save. |
| `save_on_completion` | `False` | If True, saves a checkpoint when generation completes. |
| `include_transition_to_save` | `False` | If True, includes the transition object in the checkpoint file. |

## AsyncParallelMCTS

MCTS variant for asynchronous parallel reward calculation.

| Parameter | Default | Description |
|---|---:|---|
| `*args` | required | Positional arguments passed to `MCTS`. |
| `virtual_loss` | `0.0` | This value is temporarily used as reward value until the actual reward calculation is completed. |
| `max_inflight` | required | Maximum number of reward tasks that may be in flight. Automatically set to the number of available workers in `chemtsv3-mpi`. |
| `reward_dispatcher_type` | `None` | Reward dispatcher type. Supported values are `"dummy"` and `"mpi"`. Automatically set to `"mpi"` in `chemtsv3-mpi`. |
| `check_interval` | `0.05` | Sleep interval in seconds while waiting for in-flight reward tasks. |
| `**kwargs` | `{}` | Additional arguments passed to `MCTS`. `discard_unneeded_states` defaults to False to avoid potential conflicts with other classes. |

Inherited `MCTS` parameters include `root`, `transition`, `reward`, `policy`, `filters`, `filter_reward`, `n_eval_width`, `allow_eval_overlaps`, `n_eval_iters`, `n_tries`, `cut_failed_child`, `reward_cutoff`, `reward_cutoff_warmups`, `terminal_reward`, `cut_terminal`, `avoid_duplicates`, `discard_unneeded_states`, `max_tree_depth`, `virtual_loss`, `use_dummy_reward`, `precalculated_csv_paths`, `output_dir` and all logging/checkpoint parameters. `failed_parent_reward` is disabled.

## DoubleAsyncParallelMCTS

AsyncParallelMCTS variant that also parallelizes transition expansion and rollout.

| Parameter | Default | Description |
|---|---:|---|
| `inflight_type` | `"separate"` | `"separate"` uses independent reward / transition worker limits. `"mpi"` uses shared MPI workers. |
| `reward_dispatcher_type` | `None` | Reward dispatcher type. Supported values are `"dummy"`, `"mpi"`, and `"disable"`. For transition-only parallel search, set `reward_dispatcher_type` to `"disable"`. |
| `max_reward_inflight` | required for `"separate"` | (Used in `"separate"`) Maximum number of reward tasks that may be in flight. Automatically set by `chemtsv3-mpi` if `reward_dispatcher_type` is `"mpi"`. |
| `transition_dispatcher_type` | `"thread"` for `"separate"`, `"mpi"` for `"mpi"` | Transition dispatcher type. Supports `"thread"` and `"mpi"`. |
| `max_transition_inflight` | required for `"separate"` | (Used in `"separate"`) Maximum number of transition tasks that may be in flight. Specify this explicitly when using threaded transition parallelization. |
| `max_mpi_inflight` | required for `"mpi"` | (Used in `"mpi"`) Maximum number of MPI tasks that may be in flight. Automatically set by `chemtsv3-mpi` when omitted. |
| `transition_loss` | `0.0` | Temporary reward value backpropagated while a transition job is in flight, then reverted when the transition ends. |
| `**kwargs` | `{}` | Additional arguments passed to `MCTS`. `discard_unneeded_states` defaults to False to avoid potential conflicts with async reward/transition jobs. |

Inherited `MCTS` parameters include `root`, `transition`, `reward`, `policy`, `filters`, `filter_reward`, `n_eval_width`, `allow_eval_overlaps`, `n_eval_iters`, `n_tries`, `cut_failed_child`, `reward_cutoff`, `reward_cutoff_warmups`, `terminal_reward`, `cut_terminal`, `avoid_duplicates`, `discard_unneeded_states`, `max_tree_depth`, `virtual_loss`, `use_dummy_reward`, `precalculated_csv_paths`, `output_dir` and all logging/checkpoint parameters. `failed_parent_reward` is disabled.

## HeapQueueGenerator

Generator that evaluates children and keeps candidates in a reward-prioritized heap queue.

| Parameter | Default | Description |
|---|---:|---|
| `root` | required | Root node from which generation starts. |
| `transition` | required | Transition used to expand nodes and perform rollouts. |
| `max_length` | `None` | Maximum rollout length. If omitted, `transition.max_length()` is used. |
| `**kwargs` | `{}` | Additional arguments passed to `Generator`: `reward`, `filters`, `filter_reward`, `precalculated_csv_paths`, `name`, `output_dir`, `logger`, `logging_interval`, `info_interval`, `analyze_interval`, `verbose_interval`, `save_interval`, `save_on_completion`, and `include_transition_to_save`. |

## RandomGenerator

Generator that repeatedly samples a rollout from the root node.

| Parameter | Default | Description |
|---|---:|---|
| `root` | required | Root node from which generation starts. |
| `transition` | required | Transition used to perform rollouts. |
| `max_length` | `None` | Maximum rollout length. If omitted, `transition.max_length()` is used. |
| `**kwargs` | `{}` | Additional arguments passed to `Generator`: `reward`, `filters`, `filter_reward`, `precalculated_csv_paths`, `name`, `output_dir`, `logger`, `logging_interval`, `info_interval`, `analyze_interval`, `verbose_interval`, `save_interval`, `save_on_completion`, and `include_transition_to_save`. |
