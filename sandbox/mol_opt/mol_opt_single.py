# Example: python sandbox/mol_opt/mol_opt_single.py

# Path setup / Imports
import faulthandler
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
from statistics import mean
import traceback
import optuna
from utils import conf_from_yaml, generator_from_conf

guacamol_oracle_names = ["zaleplon_mpo", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "troglitazone_rediscovery", "median1", "sitagliptin_mpo", "thiothixene_rediscovery", "deco_hop", "albuterol_similarity", "scaffold_hop", "amlodipine_mpo", "celecoxib_rediscovery", "fexofenadine_mpo", "median2", "mestranol_similarity", "perindopril_mpo", "osimertinib_mpo", "ranolazine_mpo", "valsartan_smarts"]
tdc_oracle_names = ["drd2", "gsk3b", "jnk3", "qed"]
oracle_names = guacamol_oracle_names + tdc_oracle_names

yaml_path_1 = "config/tuning/mol_opt_v3_rnn_only.yaml"
    
def reward_class_name_from_oracle_name(oracle_name: str) -> str:
    if oracle_name in tdc_oracle_names:
        return "TDCReward"
    else:
        return "GuacaMolReward"

def objective(trial):
    try:
        conf = conf_from_yaml(yaml_path_1)
        conf.setdefault("transition_args", {})
        # conf["transition_args"]["sharpness"] = trial.suggest_float("sharpness", 0.9, 1.1)
        # conf["transition_args"]["top_p"] = trial.suggest_float("top_p", 0.993, 0.997)
        conf.setdefault("policy_args", {})
        conf["policy_args"]["c"] = trial.suggest_float("c", 0.01, 0.4)
        conf["policy_args"]["best_rate"] = trial.suggest_float("best_rate", 0, 1)
        conf.setdefault("generator_args", {})
        conf["generator_args"]["n_tries"] = trial.suggest_categorical("n_tries", [1, 3])
        conf["generator_args"]["n_eval_width"] = trial.suggest_categorical("n_eval_width", [1, float("inf")])
        conf["generator_args"]["n_eval_iters"] = trial.suggest_int("n_eval_iters", 1, 3)
        
        sum_auc = 0
        for i, oracle_name in enumerate(oracle_names):
            conf["reward_class"] = reward_class_name_from_oracle_name(oracle_name)
            conf["reward_args"] = {}
            conf["reward_args"]["objective"] = oracle_name
            conf["output_dir"] = "generation_result" + os.sep + "trial_" + str(trial.number) + os.sep + oracle_name
            
            generator = generator_from_conf(conf)
            generator.logger.info("reward="+oracle_name)
            generator.logger.info(f"params={trial.params}")
            generator.generate(max_generations=10000, time_limit=conf.get("time_limit"))

            auc = generator.auc(top_k=10, max_oracle_calls=10000, finish=True)
            n_generated = len(generator.unique_keys)
            generator.logger.info(f"top_10_auc: {auc}")
            
            if n_generated < 10000:
                raise optuna.exceptions.TrialPruned()
            
            trial.set_user_attr(oracle_name, auc)
            sum_auc += auc
            intermediate_value = sum_auc
            trial.report(intermediate_value, i)
            if trial.should_prune():
                print(f"{oracle_name} Trial {trial.number} - Step {i}: intermediate_score={intermediate_value:.3f}, params={trial.params}")
                raise optuna.TrialPruned()

        trial.set_user_attr("sum_top_10_auc", sum_auc)
        print(f"{oracle_name} Trial {trial.number}: sum_top_10_auc={sum_auc:.3f}, aucs={trial.user_attrs}")
        
        return sum_auc
    except Exception as e:
        # print("Error occurred:", e)
        # traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="mol_opt_no_chain", help="Name")
    parser.add_argument("--enqueue", type=bool, default=True, help="Enqueue")
    args = parser.parse_args()
    
    name = args.name
    storage = "sqlite:///sandbox/mol_opt/optuna/" + name + ".db"
    sampler = sampler=optuna.samplers.TPESampler(multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True)
    
    if args.enqueue:
        study.enqueue_trial({"c": 0.1, "best_rate": 0.5, "n_eval_width": 1, "n_eval_iters":1, "n_tries": 1})
        
    study.optimize(objective, n_trials=200)
        
if __name__ == "__main__":
    faulthandler.enable()
    main()