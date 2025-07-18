import copy
from datetime import datetime
import inspect
import logging
import os
import random
import shutil
import time
import torch
from typing import Any
import yaml
import numpy as np
from rdkit import RDLogger
from generator import Generator
from language import Language
from node import SurrogateNode, SentenceNode, MolSentenceNode
from utils import class_from_package, make_logger

def conf_from_yaml(yaml_path: str, repo_root: str="../") -> dict[str, Any]:
    with open(os.path.join(repo_root, yaml_path)) as f:
        conf = yaml.safe_load(f)
    conf["yaml_path"] = yaml_path
    
    return conf    

def generator_from_conf(conf: dict[str, Any], repo_root: str="../", predecessor: Generator=None, n_top_keys_to_pass: int=None) -> Generator:
    conf_clone = copy.deepcopy(conf) 
    if predecessor is None:
        output_dir = os.path.join(repo_root, "sandbox", conf_clone["output_dir"], datetime.now().strftime("%m-%d_%H-%M")) + os.sep
        console_level = logging.ERROR if conf_clone.get("silent") else logging.INFO
        file_level = logging.DEBUG if conf_clone.get("debug") else logging.INFO
        csv_level = logging.ERROR if not conf_clone.get("csv_output", True) else logging.INFO
        logger = make_logger(output_dir, console_level=console_level, file_level=file_level, csv_level=csv_level)
    else:
        output_dir = predecessor._output_dir
        logger = predecessor.logger
    generator_args = conf_clone.get("generator_args", {})

    # set seed
    if "seed" in conf_clone:
        seed = conf_clone["seed"]
    else:
        seed = int(time.time()) % (2**32)
    logger.info("seed: " + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set node class
    node_class = class_from_package("node", conf_clone.get("node_class"))
    if node_class == MolSentenceNode:
        MolSentenceNode.use_canonical_smiles_as_key = conf_clone.get("use_canonical_smiles_as_key", False)

    # set transition lang
    transition_args = conf_clone.get("transition_args", {})
    for key, val in transition_args.items():
        if isinstance(val, str) and not os.path.isabs(val) and (key.endswith("_dir") or key.endswith("_path")):
            transition_args[key] = os.path.join(repo_root, val)
    transition_class = class_from_package("transition", conf_clone["transition_class"])
        
    if issubclass(node_class, SentenceNode) or "lang_path" in conf_clone:
        lang_path = conf_clone.get("lang_path")
        if lang_path is None:
            lang_name = os.path.basename(os.path.normpath(transition_args["model_dir"])) + ".lang"
            lang_path = os.path.join(transition_args["model_dir"], lang_name)
        elif not os.path.isabs(lang_path):
            lang_path = os.path.join(repo_root, lang_path)
        lang = Language.load(lang_path)
        transition_args["lang"] = lang
    elif "language_class" in conf_clone:
        language_class = class_from_package("language", conf_clone["language_class"])
        language_args = conf_clone.get("language_args", {})
        lang = language_class(**language_args)

    if "device" in inspect.signature(transition_class.__init__).parameters:
        transition_args["device"] = conf_clone.get("device")
    if "logger" in inspect.signature(transition_class.__init__).parameters:
        transition_args["logger"] = logger
    transition = transition_class(**transition_args)
    
    # set root
    root_args = {}
    if "lang" in locals() :
        root_args["lang"] = lang
    if "device" in inspect.signature(node_class.node_from_key).parameters:
        root_args["device"] = conf_clone.get("device")
        
    if n_top_keys_to_pass:
        top_keys = [key for key, _ in predecessor.top_k(k=n_top_keys_to_pass)]
        conf_clone["root"] = top_keys
    
    if type(conf_clone.get("root")) == list:
        root = SurrogateNode()
        for s in conf_clone.get("root"):
            node = node_class.node_from_key(key=s, parent=root, last_prob=1/len(conf_clone.get("root")), last_action=s, **root_args)
            root.add_child(child=node)
    else:
        root = node_class.node_from_key(key=conf_clone.get("root", ""), **root_args)

    # set reward
    if not "reward_class" in conf_clone and predecessor is not None:
        reward = predecessor.reward
    else:
        reward_class = class_from_package("reward", conf_clone.get("reward_class"))
        reward = reward_class(**conf_clone.get("reward_args", {}))
    
    # set policy
    if "policy_class" in conf_clone:
        policy_class = class_from_package("policy", conf_clone.get("policy_class"))
        policy = policy_class(**conf_clone.get("policy_args", {}))
        generator_args["policy"] = policy

    # set filters
    filter_settings = conf_clone.get("filters", [])
    filters = []
    for s in filter_settings:
        filter_class = class_from_package("filter", s.pop("filter_class"))
        filters.append(filter_class(**s))
    
    # set generator
    generator_class = class_from_package("generator", conf_clone.get("generator_class", "MCTS"))
    generator = generator_class(root=root, transition=transition, reward=reward, filters=filters, output_dir=output_dir, logger=logger, **generator_args)
    
    if predecessor:
        generator.inherit(generator)
    
    # copy yaml to the output directory
    src = os.path.join(repo_root, conf_clone["yaml_path"]) # added in conf_from_yaml
    dst = os.path.join(output_dir, os.path.basename(conf_clone["yaml_path"]))
    shutil.copy(src, dst)
    
    return generator