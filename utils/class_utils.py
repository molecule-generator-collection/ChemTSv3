import importlib
import pickle
from typing import Self

def get_class_from_class_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_class_path_from_object(obj: object) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"