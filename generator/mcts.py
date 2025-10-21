import queue
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
                 n_eval_width: int=float("inf"), allow_eval_overlaps: bool=False, n_eval_iters: int=1, n_tries: int=1, 
                 cut_failed_child: bool=False, reward_cutoff: float=None, reward_cutoff_warmups: int=None, 
                 terminal_reward: float | str="ignore", cut_terminal: bool=True, 
                 avoid_duplicates: bool=True, discard_unneeded_states: bool=None,
                 max_tree_depth=None, use_dummy_reward: bool=False,
                 name: str=None, output_dir: str=None, logger: logging.Logger=None, logging_interval: int=None, info_interval: int=100, analyze_interval: int=10000, verbose_interval: int=None, save_interval: int=None, save_on_completion: bool=False, include_transition_to_save: bool=False):
        """
        Args:
            root: The root node. Use SurrogateNode to search from multiple nodes.
            n_eval_width: The number of children to sample during the eval step. To use policy instead of sampling, set this value to 0. To evaluate all children, set this to float("inf") in Python code or .inf in YAML.
            allow_eval_overlaps: whether to allow overlap nodes when sampling eval candidates (recommended: False)
            n_eval_iters: the number of child node evaluations (rollouts for children that has_reward = False)
            n_tries: the number of attempts to obtain an unfiltered node in a single eval (should be 1 unless has_reward() can be False or filters are probabilistic)
            filter_reward: Substitute reward value used when nodes are filtered. Set to "ignore" to skip reward assignment. Use a list to specify different rewards for each filter step.
            cut_failed_child: If True, child nodes will be removed when {n_eval_iters * n_tries} evals are filtered.
            reward_cutoff: Child nodes will be removed if their reward is lower than this value.
            reward_cutoff_warmups: If specified, reward_cutoff will be inactive until {reward_cutoff_warmups} generations.
            avoid_duplicates: If True, duplicate nodes won't be added to the search tree. Should be True if the transition forms a cyclic graph. Unneeded if the tree structure of the transition graph is guranteed, and can be set to False to reduce memory usage.
            
            failed_parent_reward: (Set to -1 for v2 replication) Backpropagate this value when {n_eval_width * n_eval_iters * n_tries} evals are filtered from the node.
            cut_terminal: (Set to False for v2 replication) If True, terminal nodes will be culled, and won't be visited twice.
            terminal_reward: (Set to -1 for v2 replication) If "ignore", doesn't backpropagate anything. If float value, backpropagate specified value.
            
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
            discard_unneeded_states: If True, discards variables of nodes that will no longer be used after expansion.
            
            output_dir: Directory where the generation results and logs will be saved.
            logger: Logger instance used to record generation results.
            logging_interval: Number of generations between each logging. Overrides info_interval.
            info_interval: Number of generations between each logging of the generation result.
            analyze_interval: Number of generations between each call of analyze().
            save_interval: Number of generations between each checkpoint save.
            save_on_completion: If True, saves the checkpoint when completing the generation.
            include_transition_to_save: If True, transition will be included to the checkpoint file when saving.
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
        if cut_failed_child == True and discard_unneeded_states == True:
            raise ValueError("cut_failed_child=True with discard_unneeded_states=True is not supported.")

        self.root = root
        self.max_tree_depth = max_tree_depth
        self.policy = policy
        self.n_eval_width = n_eval_width
        self.allow_eval_overlaps = allow_eval_overlaps
        self.n_eval_iters = n_eval_iters
        self.n_tries = n_tries
        self.cut_failed_child = cut_failed_child
        self.reward_cutoff = reward_cutoff
        self.reward_cutoff_warmups = reward_cutoff_warmups or 0
        self.reward_cutoff_count = 0
        self.terminal_reward = terminal_reward
        self.cut_terminal = cut_terminal
        self.avoid_duplicates = avoid_duplicates
        if self.avoid_duplicates:
            # prevent root nodes from being visited again
            self.node_keys = set()
            self.node_keys.add(self.root.key())
            for c in self.root.children:
                self.node_keys.add(c.key())
        if discard_unneeded_states is not None:
            self.discard_unneeded_states = discard_unneeded_states
        else:
            self.discard_unneeded_states = False if cut_failed_child else True
        self.use_dummy_reward = use_dummy_reward
        self.failed_parent_reward = failed_parent_reward
        
        self.reward_queue = queue.Queue()
        self.current_parent = None
        self.parent_unfiltered_flag = False

        super().__init__(transition=transition, reward=reward, filters=filters, filter_reward=filter_reward, name=name, output_dir=output_dir, logger=logger, logging_interval=logging_interval, info_interval=info_interval, verbose_interval=verbose_interval, analyze_interval=analyze_interval, save_interval=save_interval, save_on_completion=save_on_completion, include_transition_to_save=include_transition_to_save)
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
            if self.reward_cutoff is not None and reward < self.reward_cutoff and self.reward_cutoff_warmups < self.n_generated_nodes():
                if type(objective_values[0]) != str or not self.cut_failed_child: # not filtered
                    self.reward_cutoff_count += 1
                node.leave(logger=self.logger)
        else:
            offspring = self.transition.rollout(node)
            objective_values, reward = self._get_objective_values_and_reward(offspring)
        
        self.policy.observe(child=node, objective_values=objective_values, reward=reward, is_filtered=(type(objective_values[0])==str))
        return objective_values, reward

    def _expand(self, node: Node) -> bool:
        if self.max_tree_depth is not None and node.depth > self.max_tree_depth:
            return False
        nexts = self.transition.next_nodes(node)
        if self.discard_unneeded_states:
            node.discard_unneeded_states()
        if len(nexts) == 0:
            return False
        expanded = False
        for n in nexts:
            if self.avoid_duplicates:
                key = n.key()
                if key in self.node_keys:
                    continue
                else:
                    self.node_keys.add(key)
            node.add_child(n)
            expanded = True
        return expanded

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            node.observe(0 if use_dummy_reward else value)
            node = node.parent
            
    def _generate_impl(self):
        if self.reward_queue.empty():
            if self.failed_parent_reward != "ignore" and not self.parent_unfiltered_flag:
                self._backpropagate(self.current_parent, self.failed_parent_reward, False)
            self._fill_queue()
        else:
            self._work_on_queue()
            
    def _fill_queue(self):
        node = self._selection()
        
        if not node.children and node.n != 0:
            if not self._expand(node):
                node.mark_as_terminal(cut=self.cut_terminal, logger=self.logger)
                
        if node.is_terminal():
            if self.terminal_reward != "ignore":
                self._backpropagate(node, self.terminal_reward, False)
            return

        if not node.children:
            children = [node]
        elif self.n_eval_width <= 0:
            children = [self.policy.select_child(node)]
        else:
            children = self.policy.sample_candidates(node, max_size=self.n_eval_width, replace=self.allow_eval_overlaps)
        
        self.parent_unfiltered_flag = False
        self.current_parent = node
        for child in children:
            self.reward_queue.put((child, self.n_eval_iters, self.n_tries, False)) # node for evaluation, remaining iters, remaining tries, already got unfiltered generation or not
    
    def _work_on_queue(self):
        child, iters, tries, unfiltered_flag = self.reward_queue.get()
        objective_values, reward = self._eval(child)
        
        if type(objective_values[0]) != str: # not filtered
            unfiltered_flag = True
            self.parent_unfiltered_flag = True
            self._backpropagate(child, reward, self.use_dummy_reward)
        else: # filtered
            if tries > 1:
                self.reward_queue.put((child, iters, tries-1, unfiltered_flag))
                return
            elif self.filter_reward[int(objective_values[0])] != "ignore":
                self._backpropagate(child, self.filter_reward[int(objective_values[0])], False)
                
        if iters > 1:
            self.reward_queue.put((child, iters-1, self.n_tries, unfiltered_flag))
        elif self.cut_failed_child and not unfiltered_flag:
            child.leave(logger=self.logger)

    # override
    def display_top_k_molecules(self, str2mol_func=None, k: int=15, mols_per_row=5, legends: list[str]=["order","reward"], target: str="reward", size=(200, 200)):
        if str2mol_func is not None:
            return super().display_top_k_molecules(str2mol_func, k=k, mols_per_row=mols_per_row, legends=legends, target=target, size=size)
        else:
            c = self.root.sample_child()
            if not hasattr(c, "lang"):
                raise AttributeError("Node objects don't have lang: For molecule nodes that don't use lang, specify str2mol_func.")
            str2mol_func = c.lang.sentence2mol
            return super().display_top_k_molecules(str2mol_func, k=k, mols_per_row=mols_per_row, legends=legends, target=target, size=size)
        
    # override
    def inherit(self, predecessor):
        super().inherit(predecessor)
        self.policy.on_inherit(self)
        
    # override
    def analyze(self):
        super().analyze()
        self.policy.analyze()
        if self.reward_cutoff is not None:
            self.logger.info(f"Reward cutoff count: {self.reward_cutoff_count}")