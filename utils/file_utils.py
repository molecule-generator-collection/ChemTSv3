import ast
import glob
import importlib
import inspect
import os
import pkgutil
from datetime import datetime
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
    except Exception:
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
        except ModuleNotFoundError:
            continue

        obj = getattr(module, class_name, None)
        if inspect.isclass(obj):
            return obj

    raise ImportError(f"Failed to load '{class_name}'. Please check that all required dependencies are installed and the class name is spelled correctly.")

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