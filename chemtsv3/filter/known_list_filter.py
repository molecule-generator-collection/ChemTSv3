import csv
from chemtsv3.filter import Filter
from chemtsv3.utils import resolve_path

class KnownListFilter(Filter):
    def __init__(self, list_paths: list[str]):
        self.known_nodes = {} # key: reward
        if list_paths is not None:
            self.list_paths = [resolve_path(path) for path in list_paths]
            for path in self.list_paths:
                self.register_list(path)
            
    def register_list(self, path: str):
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                if "key" not in reader.fieldnames or "reward" not in reader.fieldnames:
                    raise ValueError(f"CSV must contain 'key' and 'reward' columns: {reader.fieldnames}")

                for row in reader:
                    key = row["key"]
                    reward_str = row["reward"]

                    try:
                        reward = float(reward_str)
                    except ValueError:
                        raise ValueError(f"Invalid reward value for key={key}: {reward_str}")

                    self.known_nodes[key] = reward
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {path}")
        
    def check(self, node):
        key = node.key()
        if key in self.known_nodes:
            return self.known_nodes[key]
        else:
            return True