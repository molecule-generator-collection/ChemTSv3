# Example: python sandbox/mol_opt/tuning_chain_pred.py

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

oracle_names = ["troglitazone_rediscovery", "sitagliptin_mpo", "fexofenadine_mpo"]

yaml_path_1 = "config/mol_opt/de_novo_rnn.yaml"
yaml_path_2 = "config/tuning/mol_opt_gbga_pred.yaml"

def objective(trial):
    try:
        conf_1 = conf_from_yaml(yaml_path_1)
        
        conf_2 = conf_from_yaml(yaml_path_2)
        conf_2.setdefault("policy_args", {})
        conf_2["policy_args"].setdefault("predictor_params", {})
        conf_2["policy_args"]["alpha"] = trial.suggest_float("alpha", 0.75, 0.95)
        conf_2["policy_args"]["score_threshold"] = trial.suggest_float("score_threshold", 0.3, 0.7)
        conf_2["policy_args"]["score_calculation_interval"] = trial.suggest_categorical("score_calculation_interval", [25, 50, 100, 200, 250, 500])
        conf_2["policy_args"]["predictor_params"]["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1)
        conf_2["policy_args"]["predictor_params"]["num_leaves"] = trial.suggest_int("num_leaves", 5, 20)
        conf_2["policy_args"]["predictor_params"]["max_depth"] = trial.suggest_int("max_depth", 5, 9)
        
        n_generations_until_lead = 400
        n_keys_to_pass = 4
        
        sum_auc = 0
        for i, oracle_name in enumerate(oracle_names):
            conf_1["reward_class"] = "TDCReward"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="mol_opt_chain_pred", help="Name")
    parser.add_argument("--enqueue", type=bool, default=False, help="Enqueue")
    args = parser.parse_args()
    
    name = args.name
    storage = "sqlite:///sandbox/mol_opt/optuna/" + name + ".db"
    sampler = sampler=optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(direction="maximize", study_name=name, storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True)
        
    study.optimize(objective, n_trials=300)
        
if __name__ == "__main__":
    faulthandler.enable()
    main()