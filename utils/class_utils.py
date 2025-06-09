import importlib
import os
import re

def class_from_class_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def class_from_package(package_name: str, class_name: str):
    module = importlib.import_module(package_name)
    return getattr(module, class_name)

def class_path_from_object(obj: object) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"

def camel2snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

def add_sep(path: str) -> str:
    return path if path.endswith(os.path.sep) else path + os.path.sep