# Example (RNN): python sandbox/generation.py -c config/mcts/example.yaml
# Example (Chain): python sandbox/generation.py -c config/mcts/example_chain_1.yaml
# Example (Chain): python sandbox/generation.py -c config/personal/stest_1.yaml
# Example (Load): python sandbox/generation.py -l sandbox/generation_result/~~~/checkpoint --max_generations 100

# Path setup / Imports
import faulthandler
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
from generator import Generator
from utils import conf_from_yaml, generator_from_conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--yaml_path", type=str, help="Path to the config file (.yaml)")
    parser.add_argument("-l", "--load_dir", type=str, help="Path to the save directory (contains config.yaml and save.gtr)")
    
    parser.add_argument("--max_generations", type=int, help="Only used when loading the generator from the save.")
    parser.add_argument("-t", "--time_limit", type=int, help="Only used when loading the generator from the save.")
    
    args = parser.parse_args()
    
    yaml_path = args.yaml_path
    load_dir = args.load_dir
    
    if yaml_path is None and load_dir is None:
        raise ValueError("Specify either 'yaml_path' (-c) or 'load_dir' (-l).")
    elif yaml_path is not None and load_dir is None:
        conf = conf_from_yaml(yaml_path)
        generator = generator_from_conf(conf)
        while(yaml_path):
            generator.generate(time_limit=conf.get("time_limit"), max_generations=conf.get("max_generations"))
            if not "next_yaml_path" in conf:
                yaml_path = None
                plot_args = conf.get("plot_args", {})
                if not "save_only" in plot_args:
                    plot_args["save_only"] = True
                generator.plot(**plot_args)
                generator.analyze()
            else:
                n_top_keys_to_pass=conf.get("n_keys_to_pass", 3)
                yaml_path = conf["next_yaml_path"]
                conf = conf_from_yaml(yaml_path)
                new_generator = generator_from_conf(conf, predecessor=generator, n_top_keys_to_pass=n_top_keys_to_pass)
                generator = new_generator
                
    elif yaml_path is None and load_dir is not None:
        generator = Generator.load_dir(load_dir)
        max_generations = args.max_generations
        time_limit = args.time_limit
        generator.generate(max_generations=max_generations, time_limit=time_limit)
        generator.analyze()
        plot_args = generator.yaml_copy.get("plot_args", {})
        if not "save_only" in plot_args:
            plot_args["save_only"] = True
        generator.plot(**plot_args)
    else:
        raise ValueError("Specify one of 'yaml_path' (-c) or 'load_dir' (-l), not both.")

if __name__ == "__main__":
    faulthandler.enable()
    main()