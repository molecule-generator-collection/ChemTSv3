# Example: python sandbox/mol_opt/mol_opt_chain.py

# Path setup / Imports
import faulthandler
import gc
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

yaml_path_1 = "config/optuna/mol_opt_rnn.yaml"
yaml_path_2 = "config/optuna/mol_opt_jensen.yaml"
    
def reward_class_name_from_oracle_name(oracle_name: str) -> str:
    if oracle_name in tdc_oracle_names:
        return "TDCReward"
    else:
        return "GuacaMolReward"

def objective(trial):
    try:
        conf_1 = conf_from_yaml(yaml_path_1)
        conf_1.setdefault("transition_args", {})
        # conf_1["transition_args"]["sharpness"] = trial.suggest_float("sharpness_1", 0.9, 1.1)
        # conf_1["transition_args"]["top_p"] = trial.suggest_float("top_p_1", 0.993, 0.997)
        conf_1.setdefault("policy_args", {})
        conf_1["policy_args"]["c"] = trial.suggest_float("c_1", 0.01, 0.4)
        conf_1["policy_args"]["best_rate"] = trial.suggest_float("best_rate_1", 0, 1)
        # conf_1["policy_args"]["max_prior"] = trial.suggest_float("max_prior_1", 0, 0.8)
        # conf_1["generator_args"]["n_eval_width"] = trial.suggest_int("n_eval_width_1", 1, 40)
        conf_1.setdefault("generator_args", {})
        conf_1["generator_args"]["n_tries"] = trial.suggest_categorical("n_tries_1", [1, 3])
        
        conf_2 = conf_from_yaml(yaml_path_2)
        conf_2.setdefault("policy_args", {})
        conf_2["policy_args"]["c"] = trial.suggest_float("c_2", 0.01, 0.5)
        conf_2["policy_args"]["best_rate"] = trial.suggest_float("best_rate_2", 0.5, 1)
        conf_2["policy_args"]["pw_c"] = trial.suggest_float("pw_c_2", 1.5, 4)
        conf_2["policy_args"]["pw_alpha"] = trial.suggest_float("pw_alpha_2", 0.5, 0.7)
        
        n_generations_until_lead = trial.suggest_categorical("n_generations_until_lead", [50, 100, 150, 200, 250, 300, 400, 500, 1000])
        n_keys_to_pass = trial.suggest_categorical("n_keys_to_pass", [1, 2, 3, 4, 5, 7, 10, 15, 20])
        
        sum_auc = 0
        for i, oracle_name in enumerate(oracle_names):
            conf_1["reward_class"] = reward_class_name_from_oracle_name(oracle_name)
            conf_1["reward_args"] = {}
            conf_1["reward_args"]["objective"] = oracle_name
            conf_1["output_dir"] = "generation_result" + os.sep + "trial_" + str(trial.number) + os.sep + oracle_name
            
            generator_1 = generator_from_conf(conf_1)
            generator_1.logger.info("reward="+oracle_name)
            generator_1.logger.info(f"params={trial.params}")
            generator_1.generate(max_generations=n_generations_until_lead, time_limit=conf_1.get("time_limit"))
            
            generator_2 = generator_from_conf(conf_2, predecessor=generator_1, n_top_keys_to_pass=n_keys_to_pass)
            generator_2.generate(max_generations=10000 - n_generations_until_lead, time_limit=conf_2.get("time_limit"))

            auc = generator_2.auc(top_k=10, max_oracle_calls=10000, finish=True)
            n_generated = len(generator_2.unique_keys)
            generator_2.logger.info(f"top_10_auc: {auc}")
            del generator_1, generator_2; gc.collect()
            
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
        print("Error occurred:", e)
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

def main():
    faulthandler.enable(file=sys.stderr, all_threads=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="mol_opt_chain", help="Name")
    parser.add_argument("--enqueue", type=bool, default=True, help="Enqueue")
    args = parser.parse_args()
    
    name = args.name
    storage = "sqlite:///sandbox/mol_opt/optuna/" + name + ".db"
    sampler = sampler=optuna.samplers.TPESampler(multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True)
    
    if args.enqueue:
        study.enqueue_trial({"c_1": 0.1, "best_rate_1": 0.5, "n_tries_1": 1, "c_2": 0.25, "best_rate_2": 0.9, "pw_c_2": 3, "pw_alpha_2": 0.5, "n_generations_until_lead": 200, "n_keys_to_pass": 5})
        study.enqueue_trial({"c_1": 0.1, "best_rate_1": 0.5, "n_tries_1": 1, "c_2": 0.15, "best_rate_2": 0.95, "pw_c_2": 2, "pw_alpha_2": 0.6, "n_generations_until_lead": 200, "n_keys_to_pass": 5})
        
    study.optimize(objective, n_trials=500)
        
if __name__ == "__main__":
    main()