import base64
import gzip
import json
import logging
import math
import os
from pathlib import Path
from numbers import Integral, Real
from chemtsv3.generator import MCTS
from chemtsv3.generator.async_parallel_mcts import AsyncParallelMCTS
from chemtsv3.generator.double_async_parallel_mcts import DoubleAsyncParallelMCTS
from chemtsv3.node import Node, SentenceNode

class VisualizedMCTS(MCTS):
    """MCTS with a standalone, offline HTML tree viewer.

    Place this class first when combining it with other MCTS subclasses.
    Expanded nodes (including pruned nodes) and their states are retained.
    """
    def __init__(self, *args, output_dir: str=None, logger: logging.Logger=None, **kwargs):
        kwargs.pop("discard_unneeded_states", None)
        super().__init__(*args, discard_unneeded_states=False, output_dir=output_dir, logger=logger, **kwargs)
        self._visualized_nodes = []
        self._visualized_node_ids = {}
        self._visualized_parents = []
        self._visualized_results = {}
        stack = [(self.root, None)]
        while stack:
            node, parent_id = stack.pop()
            node_id = self._remember_node(node, parent_id)
            stack.extend((child, node_id) for child in reversed(node.children))

    @staticmethod
    def _json_value(value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, Real):
            return float(value) if math.isfinite(value) else str(value)
        if isinstance(value, (list, tuple)):
            return [VisualizedMCTS._json_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): VisualizedMCTS._json_value(v) for k, v in value.items()}
        return str(value)

    def _remember_node(self, node: Node, parent_id: int=None) -> int:
        if node not in self._visualized_node_ids:
            node_id = len(self._visualized_nodes)
            self._visualized_node_ids[node] = node_id
            self._visualized_nodes.append(node)
            if parent_id is None:
                parent_id = self._visualized_node_ids.get(node.parent)
            self._visualized_parents.append(parent_id)
        return self._visualized_node_ids[node]

    @staticmethod
    def _display_state(node: Node) -> str:
        return SentenceNode.key(node) if isinstance(node, SentenceNode) else node.key()

    @staticmethod
    def _action_label(node: Node):
        if isinstance(node, SentenceNode) and isinstance(node.last_action, Integral) and not isinstance(node.last_action, bool):
            return node.lang.id2token(int(node.last_action))
        return node.last_action

    @staticmethod
    def _pack_image(image: str | dict | None) -> str | dict | None:
        """Pack image data URIs, returning None for text labels."""
        if isinstance(image, str) and not image.startswith("data:image/"):
            return None
        prefix = "data:image/svg+xml;base64,"
        if not isinstance(image, str) or not image.startswith(prefix):
            return image
        svg = base64.b64decode(image[len(prefix):])
        compressed = gzip.compress(svg, compresslevel=6, mtime=0)
        return {"encoding": "gzip", "type": "image/svg+xml", "data": base64.b64encode(compressed).decode("ascii")}

    @staticmethod
    def _pack_data(data: dict) -> str:
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        return json.dumps({"encoding": "gzip", "data": base64.b64encode(compressed).decode("ascii")}, separators=(",", ":"))

    # override
    def on_expansion(self, node: Node):
        parent_id = self._remember_node(node)
        for child in node.children:
            self._remember_node(child, parent_id)
        super().on_expansion(node)

    # override
    def on_evaluation(self, node: Node, target: Node):
        # The target may have no parent, notably after an MPI rollout.
        target._visualized_source_id = self._remember_node(node)
        target._visualized_is_rollout = target is not node
        super().on_evaluation(node, target)

    # override
    def on_generation(self, node: Node, objective_values: list[float], reward: float):
        display = node.visualize()
        image = self._pack_image(display)
        self._visualized_results[node.key()] = {
            "node_id": getattr(node, "_visualized_source_id", None),
            "image": image,
            "is_rollout": getattr(node, "_visualized_is_rollout", False),
            "state": display if image is None else self._display_state(node),
            "type": node.__class__.__name__,
        }
        super().on_generation(node, objective_values=objective_values, reward=reward)

    # override
    def generate(self, max_generations: int=None, time_limit: float=None):
        try:
            return super().generate(max_generations=max_generations, time_limit=time_limit)
        finally:
            self.visualize()

    def visualize(self, filename: str="tree.html") -> str:
        """Write the current snapshot; call again to update it during a run."""
        self.logger.info("Making visualization files...")
        images = []
        image_ids = {}

        def image_id(image):
            image = self._pack_image(image)
            if image is None:
                return None
            key = (image["type"], image["data"]) if isinstance(image, dict) else image
            if key not in image_ids:
                image_ids[key] = len(images)
                images.append(image)
            return image_ids[key]

        active_ids = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            active_ids.add(self._visualized_node_ids[node])
            stack.extend(node.children)

        nodes = []
        for node_id, node in enumerate(self._visualized_nodes):
            key = node.key()
            display = node.visualize()
            image = self._pack_image(display)
            nodes.append({
                "id": node_id,
                "parent": self._visualized_parents[node_id],
                "key": key,
                "state": display if image is None else self._display_state(node),
                "type": node.__class__.__name__,
                "depth": node.depth,
                "n": node.n,
                "n_children": len(node.children),
                "sum_r": node.sum_r,
                "best_r": node.best_r,
                "reward": node.reward,
                "last_action": node.last_action,
                "action_label": self._action_label(node),
                "last_prob": node.last_prob,
                "terminal": node.is_terminal(),
                "frozen": node.frozen,
                "pruned": node_id not in active_ids,
                "image": image_id(image),
            })

        df = self.df()
        rows = []
        for values in df.itertuples(index=False, name=None):
            result = self._visualized_results.get(values[2], {})
            rows.append({"values": list(values), "node_id": result.get("node_id"), "image": image_id(result.get("image")), "is_rollout": result.get("is_rollout", False), "state": result.get("state", values[2]), "type": result.get("type")})
        data = self._json_value({"name": self.name(), "nodes": nodes, "columns": list(df.columns), "rows": rows, "images": images})
        # Base64 keeps keys/actions containing HTML out of the script element.
        payload = self._pack_data(data)
        template = Path(__file__).with_name("tree_viewer.html").read_text(encoding="utf-8")
        path = os.path.join(self.output_dir(), filename)
        Path(path).write_text(template.replace("/*__TREE_DATA__*/", payload), encoding="utf-8")
        self.logger.info("Search tree visualization saved to: %s", path)
        return path

class VisualizedAsyncParallelMCTS(VisualizedMCTS, AsyncParallelMCTS):
    pass

class VisualizedDoubleAsyncParallelMCTS(VisualizedMCTS, DoubleAsyncParallelMCTS):
    pass