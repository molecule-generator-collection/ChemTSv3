from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Mol, rdFingerprintGenerator
from sklearn.metrics import mean_pinball_loss
from node import Node, MolStringNode, SurrogateNode
from policy import PUCT


class PUCTWithPredictor(PUCT):
    def __init__(self, alpha=0.9, score_threshold: float=0.4, reprediction_threshold: float=0.1, n_warmup_steps=2000, batch_size=500, predictor_type="lightgbm", predictor_params=None, fp_radius=2, fp_size=512, logger= logging.Logger, **kwargs):
        """
        (EXPERIMENTAL) Unlike the parent PUCT policy, uses {predicted evaluation value + exploration term} as a score for nodes with 0 visit count, instead of inifinity.
        (IMPORTANT) n_eval_width must be set to 0 when using this policy to actually make use of it.
        
        Args:
            alpha: Quantile level for the predictor, representing the target percentile of the response variable to be estimated and used.
            score_threshold: If the recent prediction score (1 - {pinball loss} / {baseline pinball loss}) is better than this threshold, the model will be used afterwards.
            
            c: The weight of the exploration term. Higher values place more emphasis on exploration over exploitation.
            best_rate: A value between 0 and 1. The exploitation term is computed as 
                        best_rate * (best reward) + (1 - best_rate) * (average reward).
            max_prior: A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.
            pw_c: Used for progressive widening.
            pw_alpha: Used for progressive widening.
        """
        self.alpha = alpha
        self.score_threshold = score_threshold
        self.recent_score = -float("inf")
        self.reprediction_threshold = reprediction_threshold
        self.n_warmup_steps = n_warmup_steps or batch_size
        self.batch_size = batch_size
        if predictor_type == "lightgbm":
            self.predictor = LightGBMPredictor(alpha, predictor_params)
        else:
            raise ValueError("Invalid predictor type")
        
        # MolStringNode
        self.mfgen = None
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        
        self.use_model = False
        self.X_train = []
        self.X_train_new = []
        self.y_train = []
        self.y_train_new = []
        self.predicted_upper_dict = {}
        self.warned = False
        self.model_count = 0
        self.pred_count = 0
        self.reprediction_count = 0
        self.predicted_uppers = {}
        self.targets = {}
        self.model_scores = {}
        self.observe_flag = True
        super().__init__(logger=logger, **kwargs)
    
    def try_model_training(self):
        if (self.model_count == 0 and len(self.X_train_new) >= self.n_warmup_steps) or (self.model_count > 0 and len(self.X_train_new) >= self.batch_size):
            self.X_train += self.X_train_new
            self.X_train_new = []
            self.y_train += self.y_train_new
            self.y_train_new = []
            self.logger.info(f"Starting model training with {len(self.y_train)} data...")
            self.predictor.train(self.X_train, self.y_train)
            self.model_count += 1
            self.predicted_uppers[self.model_count] = []
            self.targets[self.model_count] = []
            self.logger.info("Model training finished.")
            
            for i in range(1, self.model_count):
                if len(self.targets[i]) > 5:
                    self.model_scores[i] = self.prediction_score(self.targets[i], self.predicted_uppers[i])
            
            self.logger.info("Model scores: " + ", ".join(f"{k}: {v:.3f}" for k, v in self.model_scores.items()))
            if self.calc_recent_score() > self.score_threshold:
                self.use_model = True
                self.logger.info(f"Recent score: {self.recent_score:.3f}. Surrogation will be applied.")
            else:
                self.use_model = False
                if self.recent_score != -float("inf"):
                    self.logger.info(f"Recent score: {self.recent_score:.3f}. Surrogation won't be applied.")
            
    def calc_recent_score(self):
        predicted_upper = []
        target = []
        model = self.model_count - 1
        while(len(target) < self.batch_size):
            if model == 0:
                return -float("inf")
            predicted_upper += self.predicted_uppers[model]
            target += self.targets[model]
            model -= 1
        self.recent_score = self.prediction_score(target, predicted_upper)
        return self.recent_score
        
    def prediction_score(self, target, predicted_upper):
        q_baseline = np.quantile(self.y_train, self.alpha)
        baseline_pred = np.full_like(target, q_baseline, dtype=float)

        pl_model = mean_pinball_loss(target, predicted_upper, alpha=self.alpha)
        pl_base  = mean_pinball_loss(target, baseline_pred, alpha=self.alpha)

        return 1 - pl_model / pl_base

    def observe(self, child: Node, objective_values: list[float], reward: float, is_filtered: bool):
        """Policies can update their internal state when observing the evaluation value of the node. By default, this method does nothing."""
        self.observe_flag = True
        if is_filtered or isinstance(child, SurrogateNode):
            return
        x = self.get_feature_vector(child)
        if x is None:
            if not self.warned and self.logger is not None:
                self.logger.warning("Feature vector is not defined in the Node class that is currently used. Override 'get_feature_vector' or try different policy class.")
            self.warned = True
            return
        
        self.X_train_new.append(x)
        self.y_train_new.append(reward)
        
        key = child.key()
        if key in self.predicted_upper_dict:
            model, pred = self.predicted_upper_dict[key]
            self.predicted_uppers[model].append(pred)
            self.targets[model].append(reward)
            
    def on_inherit(self, generator):
        rep = generator.root
        if rep.children:
            rep = rep.sample_child()
        node_class = rep.__class__

        for key in generator.generated_keys():
            try:
                node = node_class.node_from_key(key)
                x = self.get_feature_vector(node)
                self.X_train_new.append(x)
                self.y_train_new.append(generator.record[key]["reward"])
            except Exception as e:
                self.logger.warning(f"Generation results conversion for PUCT predictor was failed. Generation results before this message won't be used for the training of the predictor. Error details: {e}")
                return
        self.logger.info(f"Inherited generated results are converted to the training data for the PUCT predictor.")
                
    def analyze(self):
        self.logger.info(f"Number of prediction: {self.pred_count}")
        self.logger.info(f"Number of reprediction: {self.reprediction_count}")
            
    # override
    def _unvisited_node_fallback(self, node):
        self.try_model_training()
        if not self.use_model:
            if self.model_count > 0 and self.observe_flag == True:
                self.observe_flag = False
                key = node.key()
                x = self.get_feature_vector(node)
                predicted_upper_reward = self.predictor.predict_upper(x)
                self.pred_count += 1
                self.predicted_upper_dict[key] = (self.model_count, predicted_upper_reward)
                return super()._unvisited_node_fallback(node) + 1 # Should be evaluated
            else:
                return super()._unvisited_node_fallback(node)
        else: # self.use_model == True
            key = node.key()
            if key in self.predicted_upper_dict:
                model, prev_pred = self.predicted_upper_dict[key]
                if model >= self.model_count - 1 or self.model_scores[model] + self.reprediction_threshold > self.recent_score:
                    return prev_pred
                else:
                    self.reprediction_count += 1

            x = self.get_feature_vector(node)
            predicted_upper_reward = self.predictor.predict_upper(x)
            self.pred_count += 1
            self.predicted_upper_dict[key] = (self.model_count, predicted_upper_reward)
            return predicted_upper_reward + self.get_exploration_term(node)
        
    # override here to apply for other node classes
    def get_feature_vector(self, node: Node) -> np.ndarray:
        if isinstance(node, MolStringNode):
            mol = node.mol(use_cache=True)
            features = np.concatenate([self.get_rdkit_features(mol), self.calc_fingerprint(mol)])
            return features
        else:
            return None

    @staticmethod
    def get_rdkit_features(mol) -> np.ndarray:
        return np.array([desc_fn(mol) for _, desc_fn in Descriptors.descList], dtype=float)
        
    def calc_fingerprint(self, mol: Mol) -> np.ndarray:
        if self.mfgen is None:
            self.mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_size)
        fp = self.mfgen.GetFingerprint(mol)
        arr = np.zeros((self.fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
        
class UpperPredictor(ABC):
    def __init__(self, alpha=0.9, predictor_params: dict=None):
        pass
        
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict_upper(self, x: np.ndarray) -> float:
        pass
    
class LightGBMPredictor(UpperPredictor):
    def __init__(self, alpha=0.9, predictor_params: dict=None):
        self.params = predictor_params or dict(learning_rate=0.05, num_leaves=15, max_depth=6)
        self.params.setdefault("seed", 0)
        self.params.setdefault("verbose", -1)
        self.params["objective"] = "quantile"
        self.params["alpha"] = alpha
            
    def train(self, X_train, y_train):
        import lightgbm as lgb # lazy import: will be cached
        X = np.vstack(X_train).astype(np.float32)
        train_ds = lgb.Dataset(X, label=y_train)
        self.model = lgb.train(self.params, train_ds, num_boost_round=200)
    
    def predict_upper(self, x: np.ndarray) -> float:
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(X)
        return float(pred[0])