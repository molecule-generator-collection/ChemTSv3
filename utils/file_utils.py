import ast
from datetime import datetime
import glob
import importlib
import inspect
import os
from pathlib import Path
import pkgutil
import re
from types import ModuleType

def make_subdirectory(dir: str, name: str=None):
    name = name or datetime.now().strftime("%m-%d_%H-%M")
    
    base_dir = os.path.join(dir, name)
    output_dir = base_dir
    counter = 2
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}_{counter}"
        counter += 1
    output_dir += os.sep
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def class_from_class_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def class_from_package(package_name: str, class_name: str):
    # find classes in __init__.py
    try:
        pkg = importlib.import_module(package_name)
        obj = getattr(pkg, class_name)
        if inspect.isclass(obj):
            return obj
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Failed to load '{class_name}' from package '{package_name}'. "
            f"Missing dependency: '{e.name}'. Please install it."
        ) from e
    except AttributeError:
        pass

    # if not in __init__.py, try to find the class in the directory (excluding sub-directories)
    pkg = importlib.import_module(package_name)
    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if is_pkg:
            continue
        full_name = f"{package_name}.{mod_name}"

        if not contains_class(full_name, class_name):
            continue

        try:
            module: ModuleType = importlib.import_module(full_name)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Failed to load module '{full_name}' while looking for '{class_name}'. "
                f"Missing dependency: '{e.name}'. Please install it."
            ) from e

        obj = getattr(module, class_name, None)
        if inspect.isclass(obj):
            return obj

    raise ImportError(f"Failed to load '{class_name}'.")

def contains_class(module_name: str, class_name: str) -> bool:
    """Check the existence of class (while avoiding ImportError)"""
    spec = importlib.util.find_spec(module_name) # check without import
    if spec is None or not spec.origin or spec.origin.endswith((".so", ".pyd")):
        return False
    try:
        with open(spec.origin, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=spec.origin)
        for node in ast.walk(tree): # check all defined classes
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
    except Exception:
        return False
    return False

def class_path_from_object(obj: object) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"

def camel2snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

def add_sep(path: str) -> str:
    return path if path.endswith(os.path.sep) else path + os.path.sep

def find_lang_file(model_dir: str) -> str:
    """Returns: lang path"""
    lang_files = glob.glob(os.path.join(model_dir, "*.lang"))

    if len(lang_files) == 0:
        raise FileNotFoundError(f"No .lang file found in {model_dir}")
    elif len(lang_files) > 1:
        raise ValueError(f"Multiple .lang files found in {model_dir}: {lang_files}")

    return lang_files[0]

def choose_node_local_base() -> Path:
    """
    Choose a node-local base directory.

    Priority:
        1) $SLURM_TMPDIR (if set and writable)
        2) /tmp (node-local)
        3) /dev/shm (tmpfs; fast but memory-backed)

    Returns:
        Path: existing, writable directory path.
    """
    candidates = []
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        candidates.append(Path(slurm_tmp))
    candidates += [Path("/tmp"), Path("/dev/shm")]

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            if os.access(p, os.W_OK):
                return p
        except Exception:
            pass
    # Fallback: current directory
    return Path.cwd()

def setup_local_workdir(base_name: str="v3tmp", subdir: str=None, chdir: bool=False) -> Path:
    base = choose_node_local_base()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    root = base / f"{base_name}_{job_id}_{os.getpid()}"
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    if chdir:
        os.chdir(root)
    return root.resolve()

def is_running_under_slurm() -> bool:
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_SUBMIT_DIR",
        "SLURM_CLUSTER_NAME",
    ]
    return any(var in os.environ for var in slurm_vars)

def is_tmp_path(path: str | Path) -> bool:
    path = Path(path).resolve()
    return any("v3tmp" in part for part in path.parts)