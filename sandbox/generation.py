# Example: python sandbox/generation.py -c config/generation/mcts_example.yaml

# Path setup / Imports
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
from utils import conf_from_yaml, generator_from_conf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--yaml_path", type=str, help="Path to config file (.yaml)")
    args = parser.parse_args()
    
    yaml_path = args.yaml_path # Specify the yaml path

    conf = conf_from_yaml(yaml_path, repo_root)
    generator = generator_from_conf(conf, repo_root)
    generator.generate(time_limit=conf.get("time_limit"), max_generations=conf.get("max_generations"))

    plot_args = conf.get("plot_args", {})
    if not "save_only" in plot_args:
        plot_args["save_only"] = True
    generator.plot(**plot_args)
    generator.analyze()

if __name__ == "__main__":
    main()