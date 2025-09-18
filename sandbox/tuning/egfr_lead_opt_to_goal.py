# Example: python sandbox/tuning/egfr_lead_opt_to_goal.py

# Path setup / Imports
import faulthandler
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import traceback
import numpy as np
import optuna
from utils import conf_from_yaml, generator_from_conf, append_similarity_to_df, top_k_df

yaml_path = "config/tuning/egfr_lead_opt.yaml"

def objective(trial):
    try:
        conf = conf_from_yaml(yaml_path)
        conf.setdefault("transition_args", {})
        bc = trial.suggest_categorical("base_chances", ["default", "append_only"])
        conf["transition_args"]["base_chances"] = [0,0,0,0,0,0,1] if bc == "append_only" else [0.15,0.14,0.14,0.14,0.14,0.14,0.15]
        conf["policy_class"] = trial.suggest_categorical("policy", ["UCT", "PUCT"])
        conf.setdefault("policy_args", {})
        conf["policy_args"]["c"] = trial.suggest_float("c", 0.01, 1, log=True)
        conf["policy_args"]["best_rate"] = trial.suggest_float("best_rate", 0, 1)
        
        generator = generator_from_conf(conf)
        generator.generate(max_generations=conf.get("max_generations"), time_limit=conf.get("time_limit"))
        
        df = generator.df()
        append_similarity_to_df(df, goal_smiles="C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1")
        top_k = top_k_df(df, k=5, target="similarity")
        
        return np.mean(top_k["similarity"])
    except Exception as e:
        # print("Error occurred:", e)
        # traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="egfr_lead_opt_to_goal", help="Name")
    parser.add_argument("--enqueue", type=bool, default=True, help="Enqueue")
    args = parser.parse_args()
    
    name = args.name
    storage = "sqlite:///sandbox/tuning/optuna/" + name + ".db"
    sampler = sampler=optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, sampler=sampler, load_if_exists=True)
    
    if args.enqueue:
        study.enqueue_trial({"base_chances": "append_only", "policy": "UCT", "c": 0.1, "best_rate": 0})
        study.enqueue_trial({"base_chances": "default", "policy": "PUCT", "c": 0.3, "best_rate": 0})
        
    study.optimize(objective, n_trials=300)
        
if __name__ == "__main__":
    faulthandler.enable()
    main()