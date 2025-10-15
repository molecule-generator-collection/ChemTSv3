import atexit
import copy
import inspect
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
import yaml

from generator import Generator
from language import Language
from node import SurrogateNode, SentenceNode, MolSentenceNode, MolStringNode
from utils import class_from_package, make_logger, set_seed, make_subdirectory, find_lang_file, is_running_under_slurm, setup_local_workdir

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def conf_from_yaml(yaml_path: str) -> dict[str, Any]:
    with open(os.path.join(REPO_ROOT, yaml_path)) as f:
        conf = yaml.safe_load(f)
    return conf

def generator_from_conf(conf: dict[str, Any], predecessor: Generator=None, n_top_keys_to_pass: int=None) -> Generator:
    conf_clone = copy.deepcopy(conf)
    device, logger, output_dir = prepare_common_args(conf_clone, predecessor)

    save_yaml(conf, output_dir=output_dir)
    generator_args = conf_clone.get("generator_args") or {}
    set_seed(seed=conf_clone.get("seed"), logger=logger)

    # set node class (Node.lang will be set later)
    node_class = class_from_package("node", conf_clone.get("node_class"))
    if hasattr(node_class, "device"):
        node_class.device = device
    if hasattr(node_class, "logger"):
        node_class.logger = logger
    if hasattr(node_class, "output_dir"):
        node_class.output_dir = output_dir
    node_class_variables = conf_clone.get("node_class_variables") or {}
    for key, value in node_class_variables.items():
        if hasattr(node_class, key):
            if key == "device":
                logger.warning("Node class are using 'device' in 'node_class_variables', which would override 'device' in common args.")
            setattr(node_class, key, value)
        else:
            class_name = conf_clone.get("node_class")
            logger.warning(f"Node class {class_name} has no class variable {key}. Ignoring this setting.")
    # legacy (might be removed: please use node_class_variables): use_canonical_smiles_as_key can be set directly
    if node_class == MolSentenceNode and "use_canonical_smiles_as_key" in conf_clone:
        MolSentenceNode.use_canonical_smiles_as_key = conf_clone["use_canonical_smiles_as_key"]
    if issubclass(node_class, MolStringNode) and "use_canonical_smiles_as_key" in conf_clone:
        MolStringNode.use_canonical_smiles_as_key = conf_clone["use_canonical_smiles_as_key"]

    # set transition (and lang, if any)
    transition_args = conf_clone.get("transition_args") or {}
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
    if hasattr(node_class, "lang") and "lang" in locals():
        node_class.lang = lang
        
    transition = transition_class(**transition_args)
    
    # set root nodes
    if n_top_keys_to_pass:        
        top_k = predecessor.top_k(k=n_top_keys_to_pass)
        top_keys = [key for key, _ in top_k]
        top_values = [value for _, value in top_k]
        conf_clone["root"] = top_keys
    
    if type(conf_clone.get("root")) == list:
        root = SurrogateNode()
        for i, s in enumerate(conf_clone.get("root")):
            node = node_class.node_from_key(key=s, parent=root, last_prob=1/len(conf_clone.get("root")), last_action=s)
            root.add_child(child=node)
            if n_top_keys_to_pass:
                node.reward = top_values[i]
    else:
        root = node_class.node_from_key(key=conf_clone.get("root", ""))

    # set reward
    if not "reward_class" in conf_clone and predecessor is not None:
        reward = predecessor.reward
    else:
        reward_class = class_from_package("reward", conf_clone.get("reward_class"))
        reward_args = conf_clone.get("reward_args") or {}
        adjust_args(reward_class, reward_args, device, logger, output_dir)
        reward = reward_class(**reward_args)
    
    # set policy
    if "policy_class" in conf_clone:
        policy_class = class_from_package("policy", conf_clone.get("policy_class"))
        policy_args = conf_clone.get("policy_args") or {}
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
    _slurm_dir_caution = None
    if predecessor is None:
        if is_running_under_slurm() and not conf.get("direct_output_on_slurm", False):
            output_dir = make_subdirectory(os.path.join(setup_local_workdir(), conf["output_dir"]))
            output_dir_2 = make_subdirectory(os.path.join(REPO_ROOT, "sandbox", conf["output_dir"]))
            _slurm_dir_caution = f"Slurm detected: creating a temporary directory at {output_dir} to avoid I/O overhead: files will be copied to {output_dir_2} upon completion. Note that if the job is terminated before completion, files in the temporary directory will be lost. To suppress this behavior, put 'direct_output_on_slurm: true' to the top level of the YAML file."
            conf["_slurm_tmp_dir"] = True
            register_finalize_sync(output_dir, output_dir_2)
        else:
            output_dir = make_subdirectory(os.path.join(REPO_ROOT, "sandbox", conf["output_dir"]))

        console_level = logging.ERROR if conf.get("silent") else logging.INFO
        file_level = logging.DEBUG if conf.get("debug") else logging.INFO
        csv_level = logging.ERROR if not conf.get("csv_output", True) else logging.INFO
        logger = make_logger(output_dir, console_level=console_level, file_level=file_level, csv_level=csv_level)
    else:
        output_dir = predecessor._output_dir
        logger = predecessor.logger
    device = conf.get("device")
    
    if _slurm_dir_caution is not None:
        logger.warning(_slurm_dir_caution)
    
    return device, logger, output_dir

def register_finalize_sync(src_dir: str, dest_root: str):
    """
    On process exit, rsync node-local results to a shared destination.

    Args:
        src_dir: Node-local working directory to sync from.
        dest_root: Shared filesystem directory to sync into.
    """
    src_dir = Path(src_dir).resolve()
    dest_root = Path(dest_root).expanduser().resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    dest = dest_root / str(job_id)

    def _sync():
        try:
            dest.mkdir(parents=True, exist_ok=True)
            # Use rsync if available; fallback to shutil.copytree
            if shutil.which("rsync"):
                subprocess.run(
                    ["rsync", "-a", "--partial", f"{src_dir}/", f"{dest}/"],
                    check=False,
                )
            else:
                # shallow copy; overwrite if exists
                for root, _, files in os.walk(src_dir):
                    root_p = Path(root)
                    rel = root_p.relative_to(src_dir)
                    (dest / rel).mkdir(parents=True, exist_ok=True)
                    for f in files:
                        shutil.copy2(root_p / f, dest / rel / f)
        except Exception as e:
            print(f"[WARN] finalize sync failed: {e}")
    atexit.register(_sync)

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