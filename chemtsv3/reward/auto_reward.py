from abc import ABC
import numpy as np
from chemtsv3.reward import AdaptiveReward

class AutoReward(AdaptiveReward, ABC):
    """
    Adaptive reward based on automatically normalized objective values: This class converts each objective value into a desirability score in [0, 1] using a direction-aware piecewise sigmoid and aggregates the scores by a weighted geometric mean.

    Only objective_functions() needs to be implemented. (mol_objective_functions() if used together with MolReward as `class MyReward(AutoReward, MolReward)`.)
    """

    def __init__(
        self,
        weights: list[float]=None,
        minimize: list[bool] | bool=None, # default false
        saturation_values: list[float | None]=None,
        threshold_values: list[float | None]=None,
        sigmoid_saturation_target: float=0.9, default_saturation_std: float=3.0,
        sigmoid_threshold_target: float=0.1, default_threshold_std: float=3.0,
        sigmoid_mean_target: float=0.5,
        update_interval: int=50, warmup_steps: int=10,
        threshold_pass_rate_start: float=0.05,
        threshold_pass_rate_end: float=0.30,
        max_threshold_weight_boost: float=1.0,
        eps: float=1e-12, min_std: float=1e-12, min_anchor_gap_std: float=1e-6, 
        initial_reward: float=0.5,
    ):
        """
        Args:
            weights: User-specified objective weights. If None, all objectives are weighted equally.
            minimize: Whether each objective should be minimized. A scalar bool is broadcast to all objectives. Internally, minimized objectives are multiplied by -1.
            saturation_values: Values that are already sufficiently good.
            threshold_values: Values that should eventually be satisfied.
            
            sigmoid_saturation_target: Desirability score at saturation_values or the default upper anchor.
            default_saturation_std: Number of standard deviations above the mean used as the default saturation anchor when saturation_values is None.
            sigmoid_threshold_target: Desirability score at threshold_values or the default lower anchor.
            default_threshold_std: Number of standard deviations below the mean used as the default lower anchor when threshold_values is None.
            sigmoid_mean_target: Desirability score at the current generation mean.
            update_interval: Number of generated nodes between subsequent statistics updates and rebackpropagations.
            warmup_steps: Number of generated nodes before the first statistics update and rebackpropagation.
            threshold_pass_rate_start: Pass rate where threshold enforcement starts to become active.
            threshold_pass_rate_end: Pass rate where threshold enforcement becomes fully active.
            max_threshold_weight_boost: Maximum additional multiplicative weight boost for objectives whose threshold is not yet often satisfied. If set to 1.0, the maximum effective weight is 2x.
            eps: Small value used to avoid log(0) in the geometric mean.
            min_std: Minimum standard deviation used for numerical stability.
            min_anchor_gap_std: Minimum distance between the mean and sigmoid anchors, measured in units of the current standard deviation.
            initial_reward: Reward returned before the first statistics update.
        """
        self.weights = weights
        self.minimize = minimize
        self.saturation_values = saturation_values
        self.threshold_values = threshold_values

        self.sigmoid_upper_target = sigmoid_saturation_target
        self.default_upper_std = default_saturation_std
        self.sigmoid_lower_target = sigmoid_threshold_target
        self.default_lower_std = default_threshold_std
        self.sigmoid_mean_target = sigmoid_mean_target

        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.threshold_pass_rate_start = threshold_pass_rate_start
        self.threshold_pass_rate_end = threshold_pass_rate_end
        self.max_threshold_weight_boost = max_threshold_weight_boost

        self.min_std = min_std
        self.min_anchor_gap_std = min_anchor_gap_std
        self.eps = eps
        self.initial_reward = initial_reward

        self._means: np.ndarray = None
        self._stds: np.ndarray = None
        self._signs: np.ndarray = None
        self._inner_weights: np.ndarray = None
        self._effective_saturation_values: np.ndarray = None
        self._effective_threshold_values: np.ndarray = None
        self._pass_rates: np.ndarray = None
        self._threshold_alphas: np.ndarray = None
        self._tau_upper: np.ndarray = None
        self._tau_lower: np.ndarray = None
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
        Compute the weighted geometric mean of normalized objective scores.
        """
        if self._means is None:
            return float(self.initial_reward)

        values = np.asarray(objective_values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"objective_values must be 1D, but got shape={values.shape}.")
        if len(values) != len(self._means):
            raise ValueError(f"Expected {len(self._means)} objective values, but got {len(values)}.")

        y = values * self._signs
        z = np.empty_like(y, dtype=float)

        upper_mask = y >= self._means
        lower_mask = ~upper_mask

        logit_mean = self._logit(self.sigmoid_mean_target)

        z[upper_mask] = logit_mean + (y[upper_mask] - self._means[upper_mask]) / self._tau_upper[upper_mask]
        z[lower_mask] = logit_mean + (y[lower_mask] - self._means[lower_mask]) / self._tau_lower[lower_mask]

        scores = self._sigmoid(z)
        scores = np.clip(scores, self.eps, 1.0)

        inner_weights = np.asarray(self._inner_weights, dtype=float)
        weight_sum = float(np.sum(inner_weights))
        if weight_sum <= 0.0:
            raise ValueError("The sum of inner weights must be positive.")

        reward = np.exp(np.sum(inner_weights * np.log(scores)) / weight_sum)
        return float(reward)

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

        numeric_df = df[objective_names].apply(lambda col: np.array([self._to_float_or_nan(v) for v in col], dtype=float)) # should be unneeded for default generators
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

        if len(numeric_df) == 0:
            return False

        values = numeric_df.to_numpy(dtype=float)
        n_objectives = values.shape[1]

        self._ensure_arrays(n_objectives)

        y = values * self._signs

        means = np.mean(y, axis=0)
        stds = np.maximum(np.std(y, axis=0), self.min_std)

        saturation_values = self._optional_float_array(self.saturation_values, n_objectives, name="saturation_values")
        threshold_values = self._optional_float_array(self.threshold_values, n_objectives, name="threshold_values")

        saturation_y = saturation_values * self._signs
        threshold_y = threshold_values * self._signs

        has_saturation = ~np.isnan(saturation_y)
        has_threshold = ~np.isnan(threshold_y)

        upper_default = means + self.default_upper_std * stds
        lower_default = means - self.default_lower_std * stds

        effective_upper = np.where(has_saturation, saturation_y, upper_default)

        pass_rates = np.ones(n_objectives, dtype=float)
        if np.any(has_threshold):
            pass_rates[has_threshold] = np.mean(y[:, has_threshold] >= threshold_y[has_threshold], axis=0)

        threshold_alphas = np.ones(n_objectives, dtype=float)
        if np.any(has_threshold):
            raw_alpha = ((pass_rates[has_threshold] - self.threshold_pass_rate_start) / (self.threshold_pass_rate_end - self.threshold_pass_rate_start))
            threshold_alphas[has_threshold] = self._smoothstep(raw_alpha)

        effective_lower = lower_default.copy()
        if np.any(has_threshold):
            effective_lower[has_threshold] = ((1.0 - threshold_alphas[has_threshold]) * lower_default[has_threshold] + threshold_alphas[has_threshold] * threshold_y[has_threshold])

        # If user anchors are incompatible with the current distribution, keep the transformation numerically valid by minimally separating anchors.
        min_gap = self.min_anchor_gap_std * stds
        effective_upper = np.maximum(effective_upper, means + min_gap)
        effective_lower = np.minimum(effective_lower, means - min_gap)

        logit_lower = self._logit(self.sigmoid_lower_target)
        logit_mean = self._logit(self.sigmoid_mean_target)
        logit_upper = self._logit(self.sigmoid_upper_target)

        tau_upper = (effective_upper - means) / (logit_upper - logit_mean)
        tau_lower = (effective_lower - means) / (logit_lower - logit_mean)

        tau_upper = np.maximum(tau_upper, self.min_std)
        tau_lower = np.maximum(tau_lower, self.min_std)

        inner_weights = self._base_weights.copy()
        if np.any(has_threshold):
            boost = np.ones(n_objectives, dtype=float)
            boost[has_threshold] += (self.max_threshold_weight_boost * (1.0 - pass_rates[has_threshold]) * (1.0 - threshold_alphas[has_threshold]))
            inner_weights = inner_weights * boost

        self._means = means
        self._stds = stds
        self._effective_saturation_values = effective_upper
        self._effective_threshold_values = effective_lower
        self._pass_rates = pass_rates
        self._threshold_alphas = threshold_alphas
        self._tau_upper = tau_upper
        self._tau_lower = tau_lower
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
        if self.default_upper_std <= 0.0:
            raise ValueError("default_upper_std must be positive.")
        if self.default_lower_std <= 0.0:
            raise ValueError("default_lower_std must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive.")
        if not (0.0 <= self.threshold_pass_rate_start < self.threshold_pass_rate_end <= 1.0):
            raise ValueError("Require 0 <= threshold_pass_rate_start < threshold_pass_rate_end <= 1.")
        if self.max_threshold_weight_boost < 0.0:
            raise ValueError("max_threshold_weight_boost must be non-negative.")
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
        Convert an optional sequence to a float array with NaN for missing values.
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