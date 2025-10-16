from abc import ABC, abstractmethod
import logging
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Mol, rdFingerprintGenerator
from sklearn.metrics import r2_score
from node import Node, MolStringNode, SurrogateNode
from policy import PUCT


class PUCTWithPredictor(PUCT):
    def __init__(self, r2_threshold: float=0.6, n_warmup_steps=2000, batch_size=500, predictor_type="lightgbm", predictor_params=None, fp_radius=2, fp_size=1024, logger= logging.Logger, **kwargs):
        """
        (IMPORTANT) n_eval_width must be set to 0 when using this policy.
        
        Modified PUCT introduced in AlphaGo Zero. Ref: https://www.nature.com/articles/nature24270
        This version uses predictor similar to the original, unlike the vanilla PUCT class.
        Args:
            r2_threshold: If the last model (before training) had r2 score better than this threshold, the model will be used afterwards.
            
            c: The weight of the exploration term. Higher values place more emphasis on exploration over exploitation.
            best_rate: A value between 0 and 1. The exploitation term is computed as 
                        best_rate * (best reward) + (1 - best_rate) * (average reward).
            max_prior: A lower bound for the best reward. If the actual best reward is lower than this value, this value is used instead.
            pw_c: Used for progressive widening.
            pw_alpha: Used for progressive widening.
        """
        self.n_warmup_steps = n_warmup_steps or batch_size
        self.batch_size = batch_size
        self.r2_threshold = r2_threshold
        if predictor_type == "lightgbm":
            self.predictor = LightGBMPredictor(predictor_params)
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
        self.predicted_value_dict = {}
        self.warned = False
        self.trained = False
        self.n_preds = 0
        self.predicted = []
        self.target = []
        super().__init__(logger=logger, **kwargs)
    
    def try_model_training(self):
        if (not self.trained and len(self.X_train_new) >= self.n_warmup_steps) or (self.trained and len(self.X_train_new) >= self.batch_size):
            self.X_train += self.X_train_new
            self.X_train_new = []
            self.y_train += self.y_train_new
            self.y_train_new = []
            self.logger.info(f"Starting model training with {len(self.y_train)} data...")
            self.predictor.train(self.X_train, self.y_train)
            self.trained = True
            self.logger.info("Model training finished.")
            
            if len(self.target) > 0:
                r2 = r2_score(self.target, self.predicted)
                self.logger.info(f"R2 score: {r2:.3f} in the last {len(self.target)} predicted/actual value pairs.")
                if r2 > self.r2_threshold:
                    self.use_model = True
                self.predicted = []
                self.target = []

    def observe(self, child: Node, objective_values: list[float], reward: float, is_filtered: bool):
        """Policies can update their internal state when observing the evaluation value of the node. By default, this method does nothing."""
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
        if key in self.predicted_value_dict:
            self.predicted.append(self.predicted_value_dict[key])
            self.target.append(reward)
                
    def analyze(self):
        self.logger.info(f"Number of prediction: {self.n_preds}")
            
    # override
    def _unvisited_node_fallback(self, node):
        self.try_model_training()
        
        if self.trained:
            key = node.key()
            if key in self.predicted_value_dict:
                predicted_reward = self.predicted_value_dict[key]
            else:
                x = self.get_feature_vector(node)
                predicted_reward = self.predictor.predict(x)
                self.n_preds += 1
                self.predicted_value_dict[key] = predicted_reward
            
        if self.use_model: # safe to assume predicted_reward is defined
            return predicted_reward + self.get_exploration_term(node)
        else:
            return super()._unvisited_node_fallback(node)
        
    # override here to apply for other node classes
    def get_feature_vector(self, node: Node) -> np.ndarray:
        if isinstance(node, MolStringNode):
            return self.calc_fingerprint(node.mol(use_cache=True))
        else:
            return None
        
    def calc_fingerprint(self, mol: Mol) -> np.ndarray:
        if self.mfgen is None:
            self.mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_size)
        fp = self.mfgen.GetFingerprint(mol)
        arr = np.zeros((self.fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
        
class Predictor(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> float:
        pass
    
class LightGBMPredictor(Predictor):
    def __init__(self, predictor_params: dict=None):
        self.model = None
        self.params = predictor_params or dict(objective="regression", learning_rate=0.05, num_leaves=15, max_depth=6, seed=0, verbose=-1)
    
    def train(self, X_train, y_train):
        import lightgbm as lgb # lazy import: will be cached
        X = np.vstack(X_train).astype(np.float32)
        train_ds = lgb.Dataset(X, label=y_train)
        self.model = lgb.train(self.params, train_ds, num_boost_round=200)
    
    def predict(self, x: np.ndarray) -> float:
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        pred = self.model.predict(X)
        return float(pred[0]) # LightGBM returns shape (1,)