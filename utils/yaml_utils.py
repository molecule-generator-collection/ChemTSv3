from datetime import datetime
import inspect
import numpy as np
import logging
import os
import random
import shutil
import time
import torch
from typing import Any
import yaml
from rdkit import RDLogger
from generator import Generator
from language import Language
from node import SurrogateNode, SentenceNode, MolSentenceNode, MolStringNode
from utils import add_sep, class_from_package, make_logger

def conf_from_yaml(yaml_path: str, repo_root: str="../") -> dict[str, Any]:
    with open(os.path.join(repo_root, yaml_path)) as f:
        conf = yaml.safe_load(f)
    conf["yaml_path"] = yaml_path
    
    return conf    

def generator_from_conf(conf: dict[str, Any], repo_root: str="../") -> Generator:
    output_dir=os.path.join(repo_root, "sandbox", conf["output_dir"], datetime.now().strftime("%m-%d_%H-%M")) + os.sep
    file_level = logging.DEBUG if conf.get("debug") else logging.INFO
    logger = make_logger(output_dir, file_level=file_level)
    generator_args = conf.get("generator_args", {})

    # set seed
    if "seed" in conf:
        seed = conf["seed"]
    else:
        seed = int(time.time()) % (2**32)
    logger.info("seed: " + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set node class
    node_class = class_from_package("node", conf.get("node_class"))
    if node_class == MolSentenceNode:
        MolSentenceNode.use_canonical_smiles_as_key = conf.get("use_canonical_smiles_as_key", False)

    # set transition lang
    transition_args = conf.get("transition_args", {})
    if "model_dir" in transition_args:
        transition_args["model_dir"] = os.path.join(repo_root, transition_args["model_dir"])
    transition_class = class_from_package("transition", conf["transition_class"])
        
    if issubclass(node_class, SentenceNode) or "lang_path" in conf:
        lang_path = conf.get("lang_path")
        if lang_path is None:
            lang_name = os.path.basename(os.path.normpath(transition_args["model_dir"])) + ".lang"
            lang_path = add_sep(transition_args["model_dir"]) + lang_name
        lang = Language.load(lang_path)
        transition_args["lang"] = lang
    elif issubclass(node_class, MolStringNode) or "language_class" in conf:
        language_class = class_from_package("language", conf["language_class"])
        language_args = conf.get("language_args", {})
        lang = language_class(**language_args)

    if "device" in inspect.signature(transition_class.__init__).parameters:
        transition_args["device"] = conf.get("device")
    if "logger" in inspect.signature(transition_class.__init__).parameters:
        transition_args["logger"] = logger
    transition = transition_class(**transition_args)
    
    # set root
    root_args = {}
    if "lang" in locals() :
        root_args["lang"] = lang
    if "device" in inspect.signature(node_class.node_from_key).parameters:
        root_args["device"] = conf.get("device")
    
    if type(conf.get("root")) == list:
        root = SurrogateNode()
        for s in conf.get("root"):
            node = node_class.node_from_key(key=s, parent=root, last_prob=1/len(conf.get("root")), last_action=s, **root_args)
            root.add_child(action=s, child=node)
    else:
        root = node_class.node_from_key(key=conf.get("root", ""), **root_args)

    # set reward
    reward_class = class_from_package("reward", conf.get("reward_class"))
    reward = reward_class(**conf.get("reward_args", {}))
    
    # set policy
    if "policy_class" in conf:
        policy_class = class_from_package("policy", conf.get("policy_class"))
        policy = policy_class(**conf.get("policy_args", {}))
        generator_args["policy"] = policy

    # set filters
    filter_settings = conf.get("filters", [])
    filters = []
    for s in filter_settings:
        filter_class = class_from_package("filter", s.pop("filter_class"))
        filters.append(filter_class(**s))
    
    # set generator
    generator_class = class_from_package("generator", conf.get("generator_class", "MCTS"))
    generator = generator_class(root=root, transition=transition, reward=reward, filters=filters, output_dir=output_dir, logger=logger, **generator_args)
    
    # copy yaml to the output directory
    output_dir=os.path.join(repo_root, "sandbox", conf["output_dir"], datetime.now().strftime("%m-%d_%H-%M")) + os.sep
    src = os.path.join(repo_root, conf["yaml_path"]) # added in conf_from_yaml
    dst = os.path.join(output_dir, "setting.yaml")
    shutil.copy(src, dst)
    
    return generator