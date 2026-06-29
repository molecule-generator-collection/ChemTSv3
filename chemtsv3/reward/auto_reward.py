from abc import ABC
import numpy as np
from chemtsv3.reward import AdaptiveReward

class AutoReward(AdaptiveReward, ABC):
    """
    Adaptive reward based on automatically normalized objective values: This class converts each objective value into a desirability score in [0, 1] using direction-aware sigmoid and aggregates the scores by a weighted geometric mean by default.

    Only objective_functions() needs to be implemented. (mol_objective_functions() if used together with MolReward as `class MyReward(AutoReward, MolReward)`.)
    """

    def __init__(
        self,
        weights: list[float]=None,
        aggregation_type: str="geometric",
        minimize: list[bool] | bool=None, # default false
        desired_values: list[float | None]=None,
        saturation_values: list[float | None]=None,
        high_target: float=0.8, default_desired_std: float=3.0,
        low_target: float=0.1, low_target_std: float=3.0,
        sigmoid_mean_target: float=0.5,
        update_interval: int=50, warmup_steps: int=10,
        pass_rate_start: float=0.05, pass_rate_end: float=0.30, max_weight_multiplier: float=5.0,
        min_anchor_gap_std: float=1.0,
        eps: float=1e-12, min_std: float=1e-12, initial_reward: float=0.5,
    ):
        """
        Args:
            weights: User-specified objective weights. If None, all objectives are weighted equally. Objectives with weight 0 are ignored during aggregation.
            aggregation_type: How to aggregate normalized objective scores. Must be one of "geometric", "arithmetic", or "harmonic".
            minimize: Whether each objective should be minimized. A scalar bool is broadcast to all objectives. Internally, minimized objectives are multiplied by -1.
            desired_values: Values that should eventually be satisfied. None uses a temporary upper anchor based on the current mean and std.
            saturation_values: Values beyond which objectives are treated as saturated. Higher values are clipped for maximization objectives, and lower values are clipped for minimization objectives.
            high_target: Desirability score at desired_values or the temporary upper anchor.
            default_desired_std: Number of standard deviations above the mean used as the temporary upper anchor when desired_values is None.
            low_target: Desirability score at the lower anchor.
            low_target_std: Number of standard deviations below the mean used as the lower anchor.
            sigmoid_mean_target: Desirability score at the sigmoid center.
            update_interval: Number of generated nodes between subsequent statistics updates and rebackpropagations.
            warmup_steps: Number of generated nodes before the first statistics update and rebackpropagation.
            pass_rate_start: Pass rate where desired-value weight boosting starts to relax.
            pass_rate_end: Pass rate where desired-value weight boosting is fully relaxed.
            max_weight_multiplier: Maximum effective weight multiplier for objectives whose desired value is not yet often satisfied. If set to 1.0, pass-rate-based weight scaling is disabled.
            eps: Small value used to avoid log(0) in the geometric mean.
            min_std: Minimum standard deviation used for numerical stability.
            min_anchor_gap_std: Minimum distance between the sigmoid center and upper anchor, measured in units of the current standard deviation.
            initial_reward: Reward returned before the first statistics update.
        """
        self.weights = weights
        self.aggregation_type = aggregation_type
        self.minimize = minimize
        self.desired_values = desired_values
        self.saturation_values = saturation_values

        self.sigmoid_upper_target = high_target
        self.default_desired_std = default_desired_std
        self.sigmoid_lower_target = low_target
        self.low_target_std = low_target_std
        self.sigmoid_mean_target = sigmoid_mean_target

        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.pass_rate_start = pass_rate_start
        self.pass_rate_end = pass_rate_end
        self.max_weight_multiplier = max_weight_multiplier

        self.min_std = min_std
        self.min_anchor_gap_std = min_anchor_gap_std
        self.eps = eps
        self.initial_reward = initial_reward

        self._means: np.ndarray = None
        self._stds: np.ndarray = None
        self._signs: np.ndarray = None
        self._inner_weights: np.ndarray = None
        self._effective_desired_values: np.ndarray = None
        self._saturation_values: np.ndarray = None
        self._has_explicit_desired: np.ndarray = None
        self._pass_rates: np.ndarray = None
        self._pass_rate_alphas: np.ndarray = None
        self._tau_upper: np.ndarray = None
        self._tau_lower: np.ndarray = None
        self._sigmoid_centers: np.ndarray = None
        self._last_update_generation = 0

        self._validate_scalar_hyperparameters()

    def check_rebackpropagation(self, generator) -> bool:
        """
        Update normalization statistics and return whether rebackpropagation is needed.
        """
        n_generated = generator.n_generated_nodes()
        if n_generated < self.warmup_steps:
            return False

        is_first_update = self._means is None
        if not is_first_update and n_generated - self._last_update_generation < self.update_interval:
            return False

        updated = self._update_statistics(generator)
        if not updated:
            return False

        self._last_update_generation = n_generated
        return True

    def reward_from_objective_values(self, objective_values: list[float]) -> float:
        """
        Compute the weighted mean of normalized objective scores.
        """
        if self._means is None:
            return float(self.initial_reward)

        values = np.asarray(objective_values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"objective_values must be 1D, but got shape={values.shape}.")
        if len(values) != len(self._means):
            raise ValueError(f"Expected {len(self._means)} objective values, but got {len(values)}.")

        y = values * self._signs
        y = self._apply_saturation(y)
        z = np.empty_like(y, dtype=float)

        upper_mask = y >= self._sigmoid_centers
        lower_mask = ~upper_mask
        logit_mean = self._logit(self.sigmoid_mean_target)
        z[upper_mask] = logit_mean + (y[upper_mask] - self._sigmoid_centers[upper_mask]) / self._tau_upper[upper_mask]
        z[lower_mask] = logit_mean + (y[lower_mask] - self._sigmoid_centers[lower_mask]) / self._tau_lower[lower_mask]

        scores = self._sigmoid(z)
        scores = np.clip(scores, self.eps, 1.0)

        reward = self._aggregate_scores(scores, self._inner_weights)
        return float(reward)

    def _aggregate_scores(self, scores: np.ndarray, weights: np.ndarray) -> float:
        """
        Aggregate normalized scores. Zero-weight objectives are ignored.
        """
        positive_weight_mask = weights > 0.0
        active_scores = scores[positive_weight_mask]
        active_weights = weights[positive_weight_mask]

        weight_sum = float(np.sum(active_weights))
        if weight_sum <= 0.0:
            raise ValueError("The sum of inner weights must be positive.")

        if self.aggregation_type == "geometric":
            return float(np.exp(np.sum(active_weights * np.log(active_scores)) / weight_sum))
        if self.aggregation_type == "arithmetic":
            return float(np.sum(active_weights * active_scores) / weight_sum)
        if self.aggregation_type == "harmonic":
            return float(weight_sum / np.sum(active_weights / active_scores))

        raise ValueError(f"Invalid aggregation_type: {self.aggregation_type}")

    def _update_statistics(self, generator) -> bool:
        """
        Update objective statistics from generator.df().
        """
        df = generator.df()
        objective_names = self.objective_names()

        if len(objective_names) == 0:
            return False

        missing_columns = [name for name in objective_names if name not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing objective columns in generator.df(): {missing_columns}")

        n_objectives = len(objective_names)
        self._ensure_arrays(n_objectives)
        active_objective_mask = self._base_weights > 0.0
        active_objective_names = [name for name, active in zip(objective_names, active_objective_mask) if active]

        numeric_df = df[objective_names].apply(lambda col: np.array([self._to_float_or_nan(v) for v in col], dtype=float)) # should be unneeded for default generators
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any", subset=active_objective_names)

        if len(numeric_df) == 0:
            return False

        values = numeric_df.to_numpy(dtype=float)

        y = values * self._signs
        y = self._apply_saturation(y)
        y[:, ~active_objective_mask] = 0.0

        means = np.mean(y, axis=0)
        stds = np.maximum(np.std(y, axis=0), self.min_std)

        desired_values = self._optional_float_array(self.desired_values, n_objectives, name="desired_values")
        desired_y = desired_values * self._signs
        has_explicit_desired = ~np.isnan(desired_y)
        has_explicit_desired = has_explicit_desired & active_objective_mask

        upper_default = means + self.default_desired_std * stds

        effective_upper = np.where(has_explicit_desired, desired_y, upper_default)

        pass_rates = np.ones(n_objectives, dtype=float)
        if np.any(has_explicit_desired):
            pass_rates[has_explicit_desired] = np.mean(y[:, has_explicit_desired] >= desired_y[has_explicit_desired], axis=0)

        pass_rate_alphas = np.ones(n_objectives, dtype=float)
        if np.any(has_explicit_desired):
            raw_alpha = ((pass_rates[has_explicit_desired] - self.pass_rate_start) / (self.pass_rate_end - self.pass_rate_start))
            pass_rate_alphas[has_explicit_desired] = self._smoothstep(raw_alpha)

        # Keep the sigmoid center below the upper anchor
        min_gap = self.min_anchor_gap_std * stds
        sigmoid_centers = means.copy()
        sigmoid_centers = np.minimum(sigmoid_centers, effective_upper - min_gap)
        effective_lower = sigmoid_centers - self.low_target_std * stds

        logit_lower = self._logit(self.sigmoid_lower_target)
        logit_mean = self._logit(self.sigmoid_mean_target)
        logit_upper = self._logit(self.sigmoid_upper_target)

        tau_upper = (effective_upper - sigmoid_centers) / (logit_upper - logit_mean)
        tau_lower = (effective_lower - sigmoid_centers) / (logit_lower - logit_mean)

        tau_upper = np.maximum(tau_upper, self.min_std)
        tau_lower = np.maximum(tau_lower, self.min_std)

        inner_weights = self._base_weights.copy()
        if np.any(has_explicit_desired):
            boost = np.ones(n_objectives, dtype=float)
            boost[has_explicit_desired] += ((self.max_weight_multiplier - 1.0) * (1.0 - pass_rates[has_explicit_desired]) * (1.0 - pass_rate_alphas[has_explicit_desired]))
            inner_weights = inner_weights * boost

        self._means = means
        self._stds = stds
        self._effective_desired_values = effective_upper
        self._has_explicit_desired = has_explicit_desired
        self._pass_rates = pass_rates
        self._pass_rate_alphas = pass_rate_alphas
        self._tau_upper = tau_upper
        self._tau_lower = tau_lower
        self._sigmoid_centers = sigmoid_centers
        self._inner_weights = inner_weights

        return True

    def _validate_scalar_hyperparameters(self):
        """
        Validate scalar hyperparameters.
        """
        targets = [self.sigmoid_lower_target, self.sigmoid_mean_target, self.sigmoid_upper_target]
        if not all(0.0 < p < 1.0 for p in targets):
            raise ValueError("sigmoid targets must be within (0, 1).")
        if not (self.sigmoid_lower_target < self.sigmoid_mean_target < self.sigmoid_upper_target):
            raise ValueError("Require sigmoid_lower_target < sigmoid_mean_target < sigmoid_upper_target.")
        if self.aggregation_type not in {"geometric", "arithmetic", "harmonic"}:
            raise ValueError('aggregation_type must be one of "geometric", "arithmetic", or "harmonic".')
        if self.default_desired_std <= 0.0:
            raise ValueError("default_desired_std must be positive.")
        if self.low_target_std <= 0.0:
            raise ValueError("low_target_std must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive.")
        if not (0.0 <= self.pass_rate_start < self.pass_rate_end <= 1.0):
            raise ValueError("Require 0 <= pass_rate_start < pass_rate_end <= 1.")
        if self.max_weight_multiplier < 1.0:
            raise ValueError("max_weight_multiplier must be at least 1.0.")
        if self.min_std <= 0.0:
            raise ValueError("min_std must be positive.")
        if self.min_anchor_gap_std <= 0.0:
            raise ValueError("min_anchor_gap_std must be positive.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        
    def _ensure_arrays(self, n_objectives: int):
        """
        Initialize and validate objective-wise hyperparameter arrays.
        """
        if self._signs is not None:
            if len(self._signs) != n_objectives:
                raise ValueError(f"Number of objectives changed from {len(self._signs)} to {n_objectives}.")
            return

        self._base_weights = self._float_array(self.weights, n_objectives, default=1.0, name="weights")
        if np.any(self._base_weights < 0.0):
            raise ValueError("weights must be non-negative.")
        if np.sum(self._base_weights) <= 0.0:
            raise ValueError("At least one weight must be positive.")

        minimize = self._bool_array(self.minimize, n_objectives, default=False, name="minimize")
        self._signs = np.where(minimize, -1.0, 1.0)
        saturation_values = self._optional_float_array(self.saturation_values, n_objectives, name="saturation_values")
        self._saturation_values = saturation_values * self._signs

    def _apply_saturation(self, y: np.ndarray) -> np.ndarray:
        if self._saturation_values is None:
            return y

        has_saturation = ~np.isnan(self._saturation_values)
        if not np.any(has_saturation):
            return y

        return np.where(has_saturation, np.minimum(y, self._saturation_values), y)

    @staticmethod
    def _float_array(values: list[float], n: int, default: float, name: str) -> np.ndarray:
        """
        Convert a scalar-like optional sequence to a float array.
        """
        if values is None:
            return np.full(n, default, dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr), dtype=float)

        if arr.shape != (n,):
            raise ValueError(f"{name} must have length {n}, but got shape={arr.shape}.")

        return arr

    @staticmethod
    def _optional_float_array(values: list[float | None], n: int, name: str) -> np.ndarray:
        """
        Convert an optional sequence to a float array.
        """
        if values is None:
            return np.full(n, np.nan, dtype=float)

        if len(values) != n:
            raise ValueError(f"{name} must have length {n}, but got length {len(values)}.")

        return np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    @staticmethod
    def _bool_array(values: list[bool] | bool, n: int, default: bool, name: str) -> np.ndarray:
        """
        Convert a scalar bool or bool sequence to a bool array.
        """
        if values is None:
            return np.full(n, default, dtype=bool)

        if isinstance(values, bool):
            return np.full(n, values, dtype=bool)

        arr = np.asarray(values, dtype=bool)
        if arr.shape != (n,):
            raise ValueError(f"{name} must have length {n}, but got shape={arr.shape}.")

        return arr

    @staticmethod
    def _to_float_or_nan(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _logit(p: float) -> float:
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _smoothstep(x: np.ndarray) -> np.ndarray:
        """
        Smoothly map values from [0, 1] to [0, 1].
        """
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
