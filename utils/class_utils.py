import importlib

def get_class_from_str(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class Serializable():
    def __getstate__(self):
        return {"attributes": {k: self._serialize(v) for k, v in self.__dict__.items()}}

    def __setstate__(self, state):
        self.__dict__.update({k: self._deserialize(v) for k, v in state["attributes"].items()})

    def _serialize(self, obj):
        if hasattr(obj, "__getstate__"):
            return obj.__getstate__()
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._serialize(v) for v in obj)
        elif isinstance(obj, set):
            return {"__set__": [self._serialize(v) for v in obj]}
        else:
            return obj

    def _deserialize(self, data):
        if isinstance(data, dict) and "attributes" in data:
            obj = self.__class__.__new__(self.__class__)
            obj.__setstate__(data)
            return obj
        elif isinstance(data, dict) and "__set__" in data:
            return set(self._deserialize(v) for v in data["__set__"])
        elif isinstance(data, dict):
            return {k: self._deserialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._deserialize(v) for v in data)
        else:
            return data