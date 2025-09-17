import copy
import inspect
import logging
import os
from typing import Any
import yaml
from generator import Generator
from language import Language
from node import SurrogateNode, SentenceNode, MolSentenceNode, MolStringNode
from utils import class_from_package, make_logger, set_seed, make_subdirectory, find_lang_file

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def conf_from_yaml(yaml_path: str) -> dict[str, Any]:
    with open(os.path.join(REPO_ROOT, yaml_path)) as f:
        conf = yaml.safe_load(f)
    return conf

def generator_from_conf(conf: dict[str, Any], predecessor: Generator=None, n_top_keys_to_pass: int=None) -> Generator:
    conf_clone = copy.deepcopy(conf)
    device, logger, output_dir = prepare_common_args(conf_clone, predecessor)

    save_yaml(conf, output_dir=output_dir)
    generator_args = conf_clone.get("generator_args", {})
    set_seed(seed=conf_clone.get("seed"), logger=logger)

    # set node class
    node_class = class_from_package("node", conf_clone.get("node_class"))
    if node_class == MolSentenceNode:
        MolSentenceNode.use_canonical_smiles_as_key = conf_clone.get("use_canonical_smiles_as_key", False)
    if issubclass(node_class, MolStringNode):
        MolStringNode.use_canonical_smiles_as_key = conf_clone.get("use_canonical_smiles_as_key", False)

    # set transition (and lang, if any)
    transition_args = conf_clone.get("transition_args", {})
    transition_class = class_from_package("transition", conf_clone["transition_class"])
    adjust_args(transition_class, transition_args, device, logger, output_dir)
    
    if "filters" in transition_args: # For TemplateTransition
        transition_args["filters"] = construct_filters(transition_args.get("filters",[]), device, logger, output_dir)
        
    if issubclass(node_class, SentenceNode) or "lang_path" in conf_clone:
        lang_path = conf_clone.get("lang_path")
        if lang_path is None:
            lang_path = find_lang_file(transition_args["model_dir"])
        elif not os.path.isabs(lang_path):
            lang_path = os.path.join(REPO_ROOT, lang_path)
        lang = Language.load(lang_path)
        transition_args["lang"] = lang
    elif "language_class" in conf_clone:
        language_class = class_from_package("language", conf_clone["language_class"])
        language_args = conf_clone.get("language_args", {})
        lang = language_class(**language_args)
        
    transition = transition_class(**transition_args)
    
    # set root
    root_args = {}
    if "lang" in locals() :
        root_args["lang"] = lang
    if "device" in inspect.signature(node_class.node_from_key).parameters:
        root_args["device"] = device
    if "logger" in inspect.signature(node_class.node_from_key).parameters:
        root_args["logger"] = logger
        
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
        reward_args = conf_clone.get("reward_args", {})
        adjust_args(reward_class, reward_args, device, logger, output_dir)
        reward = reward_class(**reward_args)
    
    # set policy
    if "policy_class" in conf_clone:
        policy_class = class_from_package("policy", conf_clone.get("policy_class"))
        policy_args = conf_clone.get("policy_args", {})
        adjust_args(policy_class, policy_args, device, logger, output_dir)
        policy = policy_class(**policy_args)
        generator_args["policy"] = policy

    # set filters
    if not "filters" in conf_clone and predecessor is not None:
        filters = predecessor.filters
    else:
        filter_settings = conf_clone.get("filters", [])
        filters = construct_filters(filter_settings, device, logger, output_dir)
    
    # set generator
    generator_class = class_from_package("generator", conf_clone.get("generator_class", "MCTS"))
    adjust_args(generator_class, generator_args, device, logger, output_dir)
    generator = generator_class(root=root, transition=transition, reward=reward, filters=filters, **generator_args)
    generator._set_yaml_copy(conf)
    
    if predecessor:
        generator.inherit(predecessor)
    
    return generator

def adjust_args(cl, args_dict: dict, device: str, logger: logging.Logger, output_dir: str):
    adjust_path_args(args_dict)
    set_common_args(cl, args_dict, device, logger, output_dir)

def set_common_args(cl, args_dict: dict, device: str, logger: logging.Logger, output_dir: str):
    if "device" in inspect.signature(cl.__init__).parameters:
        args_dict["device"] = device
    if "logger" in inspect.signature(cl.__init__).parameters:
        args_dict["logger"] = logger
    if "output_dir" in inspect.signature(cl.__init__).parameters:
        args_dict["output_dir"] = output_dir
        
def adjust_path_args(args_dict: dict):
    for key, val in args_dict.items():
        if isinstance(val, str) and not os.path.isabs(val) and (key.endswith("_dir") or key.endswith("_path")):
            args_dict[key] = os.path.join(REPO_ROOT, val)

def prepare_common_args(conf: dict, predecessor: Generator=None) -> tuple[str, logging.Logger, str]:
    if predecessor is None:
        output_dir = make_subdirectory(os.path.join(REPO_ROOT, "sandbox", conf["output_dir"]))

        console_level = logging.ERROR if conf.get("silent") else logging.INFO
        file_level = logging.DEBUG if conf.get("debug") else logging.INFO
        csv_level = logging.ERROR if not conf.get("csv_output", True) else logging.INFO
        logger = make_logger(output_dir, console_level=console_level, file_level=file_level, csv_level=csv_level)
    else:
        output_dir = predecessor._output_dir
        logger = predecessor.logger
    device = conf.get("device")
    
    return device, logger, output_dir

def construct_filters(filter_settings, device, logger, output_dir):
    if filter_settings is None:
        return []
    filters = []
    for s in filter_settings:
        filter_class = class_from_package("filter", s.pop("filter_class"))
        adjust_args(filter_class, s, device, logger, output_dir)
        filters.append(filter_class(**s))
    return filters

def save_yaml(conf: dict, output_dir: str, name: str="config.yaml", overwrite: bool=False):
    path = os.path.join(output_dir, name)
    name, ext = os.path.splitext(name)

    # prevent overwriting
    if not overwrite:
        counter = 2
        while os.path.exists(path):
            path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            counter += 1

    with open(path, "w") as f:
        yaml.dump(conf, f, sort_keys=False)

    return path