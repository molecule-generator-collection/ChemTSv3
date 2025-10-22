# Example (Chain): python sandbox/mol_opt/mol_opt_final.py --method chain
# Example (No chain / RNN only): python sandbox/mol_opt/mol_opt_final.py --method no_chain
# Example (Chain with predictor): python sandbox/mol_opt/mol_opt_final.py --method chain_with_pred
# Example (ChemTSv2 replication): python sandbox/mol_opt/mol_opt_final.py --method v2_replication

# Path setup / Imports
import faulthandler
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
from statistics import mean
from utils import conf_from_yaml, generator_from_conf

oracle_names = ["deco_hop", "troglitazone_rediscovery", "ranolazine_mpo", "celecoxib_rediscovery", "drd2", "gsk3b", "fexofenadine_mpo", "thiothixene_rediscovery", "jnk3", "scaffold_hop", "zaleplon_mpo", "isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl", "median1", "sitagliptin_mpo", "albuterol_similarity", "amlodipine_mpo", "median2", "mestranol_similarity", "perindopril_mpo", "osimertinib_mpo", "qed", "valsartan_smarts"]
    
def test_chain(oracle_name: str, seed: int, yaml_path_1: str, yaml_path_2: str) -> float:
    conf_1 = conf_from_yaml(yaml_path_1)
    conf_1["seed"] = seed
    conf_1["reward_class"] = "TDCReward"
    conf_1["reward_args"] = {}
    conf_1["reward_args"]["objective"] = oracle_name
    conf_1["output_dir"] = "generation_result" + os.sep + "seed_" + str(seed) + os.sep + oracle_name
    generator_1 = generator_from_conf(conf_1)
    generator_1.generate(max_generations=conf_1.get("max_generations"), time_limit=conf_1.get("time_limit"))

    conf_2 = conf_from_yaml(yaml_path_2)
    conf_2["seed"] = seed
    generator_2 = generator_from_conf(conf_2, predecessor=generator_1, n_top_keys_to_pass=conf_1.get("n_keys_to_pass", 3))
    generator_2.generate(max_generations=conf_2.get("max_generations"), time_limit=conf_2.get("time_limit"))

    # generator_2.plot(**conf_2.get("plot_args", {}))
    generator_2.analyze()
    result = generator_2.auc(top_k=10, max_oracle_calls=10000, finish=True)
    
    return result

def test_single(oracle_name: str, seed: int, yaml_path: str) -> float:
    conf = conf_from_yaml(yaml_path)
    conf["seed"] = seed
    conf["reward_class"] = "TDCReward"
    conf["reward_args"] = {}
    conf["reward_args"]["objective"] = oracle_name
    conf["output_dir"] = "generation_result" + os.sep + "seed_" + str(seed) + os.sep + oracle_name
    generator = generator_from_conf(conf)
    generator.generate(max_generations=conf.get("max_generations"), time_limit=conf.get("time_limit"))
    
    # generator.plot(**conf.get("plot_args", {}))
    generator.analyze()
    result = generator.auc(top_k=10, max_oracle_calls=10000, finish=True)
    
    return result

def test_objective(oracle_name: str, seed: int, method: str="chain") -> float:
    if method == "chain":
        return test_chain(oracle_name, seed, "config/mol_opt/de_novo_rnn.yaml", "config/mol_opt/lead_gbga.yaml")
    elif method == "chain_with_pred":
        return test_chain(oracle_name, seed, "config/mol_opt/de_novo_rnn.yaml", "config/mol_opt/lead_gbga_pred.yaml")
    elif method == "no_chain":
        return test_single(oracle_name, seed, "config/mol_opt/no_chain.yaml")
    elif method == "v2_replication":
        return test_single(oracle_name, seed, "config/mol_opt/v2_replication.yaml")
    else:
        raise ValueError("Invalid method name.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["chain", "chain_with_pred", "no_chain", "v2_replication"], default="chain", help="Generation mode")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of seeds (trials)")
    args = parser.parse_args()
    
    results = {}
    for oracle_name in oracle_names:
        results[oracle_name] = []
    results["sum"] = []    

    # test
    for i in range(args.n_trials):
        print(f"----------- seed: {i} -----------")
        sum = 0
        for oracle_name in oracle_names:
            score = test_objective(oracle_name, seed=i, method=args.method)
            print(oracle_name, score)
            results[oracle_name].append(score)
            sum += score
        results["sum"].append(sum)
        
    # show results
    print("sum", mean(results["sum"]), results["sum"])
    for oracle_name in oracle_names:
        print(oracle_name, mean(results[oracle_name]), results[oracle_name])
        
if __name__ == "__main__":
    faulthandler.enable()
    main()