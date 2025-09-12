# Example: python sandbox/tuning/egfr_lead_opt.py

# Path setup / Imports
import faulthandler
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import traceback
import optuna
from utils import conf_from_yaml, generator_from_conf

yaml_path = "config/optuna/egfr_lead_opt.yaml"

def txt2list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines

phase4 = txt2list(repo_root + "/data/dyramo/phase4.txt")
phase3 = txt2list(repo_root + "/data/dyramo/phase3.txt")
phase2 = txt2list(repo_root + "/data/dyramo/phase2.txt")
phase1 = txt2list(repo_root + "/data/dyramo/phase1.txt")
preclinical = txt2list(repo_root + "/data/dyramo/preclinical_active.txt")

def objective(trial):
    try:
        conf = conf_from_yaml(yaml_path)
        conf.setdefault("policy_args", {})
        conf["policy_args"]["c"] = trial.suggest_float("c", 0.1, 10, log=True)
        conf["policy_args"]["best_rate"] = trial.suggest_float("best_rate", 0, 1)
        conf["policy_args"]["epsilon"] = trial.suggest_categorical("epsilon", [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
        conf.setdefault("generator_args", {})
        conf["generator_args"]["n_eval_width"] = trial.suggest_categorical("n_eval_width", [1, float("inf")])
        
        generator = generator_from_conf(conf)
        generator.generate(max_generations=conf.get("max_generations"), time_limit=conf.get("time_limit"))
        
        df = generator.df()
        
        np4 = len(list(set(df["key"]) & set(phase4)))
        np3 = len(list(set(df["key"]) & set(phase3)))
        np2 = len(list(set(df["key"]) & set(phase2)))
        np1 = len(list(set(df["key"]) & set(phase1)))
        npc = len(list(set(df["key"]) & set(preclinical)))

        print(f"Trial {trial.number}: preclinical={npc}, phase1={np1}, phase2={np2}, phase3={np3}, phase4={np4}")
        
        return npc + 10*np1 + 15*np2 + 20*np3 + 30*np4
    except Exception as e:
        # print("Error occurred:", e)
        # traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="egfr_lead_opt", help="Name")
    parser.add_argument("--enqueue", type=bool, default=True, help="Enqueue")
    args = parser.parse_args()
    
    name = args.name
    storage = "sqlite:///sandbox/mol_opt/optuna/" + name + ".db"
    sampler = sampler=optuna.samplers.TPESampler(multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True)
    
    if args.enqueue:
        study.enqueue_trial({"c": 3, "best_rate": 0.2, "epsilon": 0.05, "n_eval_width": 1})
        
    study.optimize(objective, n_trials=10000)
        
if __name__ == "__main__":
    faulthandler.enable()
    main()