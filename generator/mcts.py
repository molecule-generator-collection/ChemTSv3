import logging
from filter import Filter
from generator import Generator
from node import Node
from policy import Policy, UCT
from reward import Reward, LogPReward
from transition import Transition

class MCTS(Generator):
    """Perform MCTS to maximize the reward."""
    def __init__(self, root: Node, transition: Transition, reward: Reward=LogPReward(), policy: Policy=UCT(), filters: list[Filter]=None, 
                 filter_reward: float | str | list=0, failed_parent_reward: float | str="ignore", 
                 eval_width: int=1, allow_eval_overlaps: bool=False, n_evals: int=1, n_tries: int=1, 
                 cut_failed_child: bool=False, reward_cutoff: float=None, 
                 terminal_reward: float | str="ignore", cut_terminal: bool=True, 
                 avoid_duplicates: bool=False, 
                 max_tree_depth=None, use_dummy_reward: bool=False, return_nodes: bool=False, 
                 name: str=None, output_dir: str=None, logger: logging.Logger=None, info_interval: int=100):
        """
        Args:
            root: The root node. Use SurrogateNode to search from multiple nodes.
            eval_width: The number of children to sample during eval step. To use policy, set this value to 0. To evaluate all children, set this to float("inf") in Python code or .inf in YAML.
            allow_eval_overlaps: whether to allow overlap nodes when sampling eval candidates (recommended: False)
            n_evals: the number of child node evaluations (rollouts for children that has_reward = False)
            n_tries: the number of attempts to obtain an unfiltered node in a single eval (should be 1 unless has_reward() can be False or filters are probabilistic)
            filter_reward: Substitute reward value used when nodes are filtered. Set to "ignore" to skip reward assignment. Use a list to specify different rewards for each filter step.
            cut_failed_child: If True, child nodes will be removed when {n_evals * n_tries} evals are filtered.
            reward_cutoff: Child nodes will be removed if their reward is lower than this value.
            avoid_duplicates: If True, duplicate nodes won't be added to the search tree. Should be True if the transition forms a cyclic graph.
            
            failed_parent_reward: (Set to -1 for v2 replication) Backpropagate this value when {eval_width * n_evals * n_tries} evals are filtered from the node.
            cut_terminal: (Set to False for v2 replication) If True, terminal nodes will be culled, and won't be visited twice.
            terminal_reward: (Set to -1 for v2 replication) If "ignore", doesn't backpropagate anything. If float value, backpropagate specified value.
            
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
            return_nodes: If set to True, generate() returns generated nodes as a list.
            
            output_dir: Directory where the generation results and logs will be saved.
            logger: Logger instance used to record generation results.
            info_interval: Number of generations between each logging of the generation result.
        """

        if not isinstance(terminal_reward, (float, int)) and terminal_reward not in ("ignore", "reward"):
            raise ValueError("terminal_reward must be one of the following: float value, 'ignore', or 'reward'.")
        if terminal_reward == "ignore" and not cut_terminal:
            raise ValueError("Set cut_terminal to True, or set terminal_reward to something else.")
        if cut_failed_child and allow_eval_overlaps:
            raise ValueError("Set one of these values to False: cut_failed_child or allow_eval_overlaps.")
        if type(filter_reward) == list and len(filter_reward) != len(filters):
            raise ValueError("The size of list input for filter_reward should match the number of filters.")
        if type(filter_reward) == list and n_tries != 1:
            raise ValueError("List input for filter_reward is not supported on n_tries > 1.")

        self.root = root
        self.max_tree_depth = max_tree_depth or transition.max_length()
        self.policy = policy
        self.eval_width = eval_width
        self.allow_eval_overlaps = allow_eval_overlaps
        self.n_evals = n_evals
        self.n_tries = n_tries
        self.cut_failed_child = cut_failed_child
        self.reward_cutoff = reward_cutoff
        self.terminal_reward = terminal_reward
        self.cut_terminal = cut_terminal
        self.avoid_duplicates = avoid_duplicates
        if self.avoid_duplicates:
            self.node_keys = set()
        self.use_dummy_reward = use_dummy_reward
        self.failed_parent_reward = failed_parent_reward
        super().__init__(transition=transition, reward=reward, filters=filters, filter_reward=filter_reward, return_nodes=return_nodes, name=name, output_dir=output_dir, logger=logger, info_interval=info_interval)
        self.root.n = 1
        
    def _selection(self) -> Node:
        node = self.root
        if not self.root.children and (self.root.n > 1 or self.root.is_terminal()):
            self.logger.info("Search tree exhausted.")
            raise SystemExit
        while node.children:
            node = self.policy.select_child(node)
        return node
    
    def _eval(self, node: Node):
        if node.has_reward():
            objective_values, reward = self._get_objective_values_and_reward(node)
            node.reward = reward
            if self.reward_cutoff is not None and reward < self.reward_cutoff:
                node.leave(logger=self.logger)
        else:
            offspring = self.transition.rollout(node)
            objective_values, reward = self._get_objective_values_and_reward(offspring)
        
        self.policy.observe(child=node, objective_values=objective_values, reward=reward)
        return objective_values, reward

    def _expand(self, node: Node) -> bool:
        transitions = self.transition.transitions(node)
        if len(transitions) == 0:
            return False
        expanded = False
        _, nodes, _ = zip(*transitions)
        for n in nodes:
            if self.avoid_duplicates:
                if n.key() in self.node_keys:
                    continue
                else:
                    self.node_keys.add(n.key())
            node.add_child(n)
            expanded = True
        return expanded

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            node.observe(0 if use_dummy_reward else value)
            node = node.parent

    # implement
    def _generate_impl(self):
        node = self._selection()
        
        if node.depth > self.max_tree_depth:
            node.mark_as_terminal(cut=self.cut_terminal, logger=self.logger)  
        elif not node.children and node.n != 0:
            if not self._expand(node):
                node.mark_as_terminal(cut=self.cut_terminal, logger=self.logger)
                
        if node.is_terminal():
            if self.terminal_reward != "ignore":
                self._backpropagate(node, self.terminal_reward, False)
            return

        if not node.children:
            children = [node]
        elif self.eval_width <= 0:
            children = [self.policy.select_child(node)]
        else:
            children = node.sample_children(max_size=self.eval_width, replace=self.allow_eval_overlaps)
        
        parent_got_unfiltered_node = False
        for child in children:
            child_got_unfiltered_node = False
            for _ in range(self.n_evals):
                for _ in range(self.n_tries):
                    objective_values, reward = self._eval(child) # returns the child itself if terminal
                    if type(objective_values[0]) != str: # not filtered
                        break
                if type(objective_values[0]) != str: # not filtered
                    child_got_unfiltered_node = parent_got_unfiltered_node = True
                    self._backpropagate(child, reward, self.use_dummy_reward)
                elif self.filter_reward[int(objective_values[0])] != "ignore":
                    self._backpropagate(child, self.filter_reward[int(objective_values[0])], False)
            if self.cut_failed_child and not child_got_unfiltered_node:
                child.leave(logger=self.logger)
        if self.failed_parent_reward != "ignore" and not parent_got_unfiltered_node:
            self._backpropagate(node, self.failed_parent_reward, False)
            self.logger.debug("All evals failed from: " + str(node))