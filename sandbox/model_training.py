#Example: python sandbox/model_training.py -c config/training/train_rnn_smiles.yaml

# Path setup / Imports
import argparse
import faulthandler
import sys
import yaml
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from transition import GPT2Transition, RNNTransition
from utils import save_yaml

def save_conf_and_lang(output_dir, conf, lang):
    for root, dirs, files in os.walk(output_dir):
        if not dirs: # bottom
            save_yaml(conf=conf, output_dir=root)
            lang.save(os.path.join(root, "language.lang"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--yaml_path", type=str, help="Path to the config file (.yaml)")
    
    args = parser.parse_args()
    
    yaml_path = args.yaml_path
    with open(os.path.join(repo_root, yaml_path)) as f:
        conf = yaml.safe_load(f)
    output_dir = os.path.join(repo_root, conf.get("output_dir"))
    
    model_type = conf.get("model_type")
    if model_type is None:
        raise ValueError("Specify model_type ('RNN' or 'GPT2') in yaml.")    
    elif model_type == "RNN":
        conf.pop("model_type")
        conf.pop("output_dir")
        model, best_state_dict, lang = RNNTransition.train_rnn_from_conf(conf)
        model.save(os.path.join(output_dir, "last"))
        model.load_state_dict(best_state_dict)
        model.save(os.path.join(output_dir, "best"))
    else: # GPT2
        model, trainer, lang = GPT2Transition.train_gpt2_from_conf(conf)
        
    save_conf_and_lang(output_dir, conf, lang)
    print("Training finished / Model saved.")

if __name__ == "__main__":
    faulthandler.enable()
    main()