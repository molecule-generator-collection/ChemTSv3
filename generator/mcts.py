import logging
from filter import Filter
from generator import Generator
from node import Node
from policy import Policy, UCT
from reward import Reward, LogPReward
from transition import Transition

class MCTS(Generator):
    def __init__(self, root: Node, transition: Transition, max_tree_depth=None, output_dir=None, name=None, reward: Reward=LogPReward(), policy: Policy=UCT(), filters: list[Filter]=None, filtered_reward: float | str | list=0, all_filtered_reward: float | str="ignore", rollout_width: int=1, allow_rollout_overlaps: bool=False, n_rollouts: int=1, n_tries: int =1, remove_failed_child: bool=False, terminal_reward: float | str="ignore", freeze_terminal: bool=True, check_loop: bool=False, use_dummy_reward: bool=False, logger: logging.Logger=None, info_interval: int=100):
        """
        Perform MCTS to maximize the reward.

        Args:
            root: The root node. Use SurrogateNode to search from multiple nodes.
            rollout_width: The number of children to sample during rollout. To perform rollouts for all children, set this to a value higher than the number of tokens.
            allow_rollout_overlaps: whether to allow overlap nodes when sampling rollout candidates (recommended: False)
            n_rollouts: the number of rollouts from one child node
            n_tries: the number of attempts to obtain an unfiltered node in a single rollout
            filtered_reward: Backpropagate this value when {n_tries} rollouts are filtered from the child. Set "ignore" not to backpropagate. Use list input if you want to set different rewards for each filter step.
            check_loop: If True, duplicate nodes won't be added to the search tree.
            use_dummy_reward: If True, backpropagate value is fixed to 0. (still calculates rewards and objective values)
            
            --- The following variables are provided for ChemTSv2 replication, and are generally recommended to leave at their default values. ---
            all_filtered_reward: Backpropagate this value when {rollout_width * n_rollouts * n_tries} rollouts are filtered from the node.
            remove_failed_child: If True, child nodes will be removed when {n_rollouts * n_tries} rollouts are filtered.
            freeze_terminal: If True, terminal node won't be visited twice.
            terminal_reward: If "ignore", doesn't backpropagate anything. If float value, backpropagate specified value.
        """

        if not isinstance(terminal_reward, (float, int)) and terminal_reward not in ("ignore", "reward"):
            raise ValueError("terminal_reward must be one of the following: float value, 'ignore', or 'reward'.")
        if terminal_reward == "ignore" and not freeze_terminal:
            raise ValueError("Set freeze_terminal to True, or set terminal_reward to something else.")
        if remove_failed_child and allow_rollout_overlaps:
            raise ValueError("Set one of these values to False: remove_failed_child or allow_rollout_overlaps.")
        if type(filtered_reward) == list and len(filtered_reward) != len(filters):
            raise ValueError("The size of list input for filtered_reward should match the number of filters.")
        if type(filtered_reward) == list and n_tries != 1:
            raise ValueError("List input for filtered_reward is not supported on n_tries > 1.")

        self.root = root
        self.max_tree_depth = max_tree_depth or transition.max_length()
        self.policy = policy
        self.rollout_width = rollout_width
        self.allow_rollout_overlaps = allow_rollout_overlaps
        self.n_rollouts = n_rollouts
        self.n_tries = n_tries
        self.remove_failed_child = remove_failed_child
        self.terminal_reward = terminal_reward
        self.freeze_terminal = freeze_terminal
        self.check_loop = check_loop
        if self.check_loop:
            self.node_keys = set()
        self.use_dummy_reward = use_dummy_reward
        self.all_filtered_reward = all_filtered_reward
        super().__init__(transition=transition, output_dir=output_dir, name=name, reward=reward, filters=filters, filtered_reward=filtered_reward, logger=logger, info_interval=info_interval)
        self.root.n = 1
        
    def _selection(self) -> Node:
        node = self.root
        while node.children:
            node = self.policy.select_child(node)
            if node.sum_r == -float("inf"): # already exhausted every terminal under this node
                self.logger.debug("Exhausted every terminal under: " + str(node.parent))
                if node.parent == self.root:
                    self.logger.info("Search tree exhausted.")
                    raise SystemExit
                node.parent.sum_r = -float("inf")
                node = self.root
        return node

    def _expand(self, node: Node) -> bool:
        transitions = self.transition.transitions_with_probs(node)
        if len(transitions) == 0:
            return False
        expanded = False
        actions, nodes, _ = zip(*transitions)
        for a, n in zip(actions, nodes):
            if self.check_loop:
                if n.key() in self.node_keys:
                    continue
                else:
                    self.node_keys.add(n.key())
            node.add_child(a, n)
            expanded = True
        return expanded
    
    def _eval(self, node: Node):
        if node.has_reward():
            return self.get_objective_values_and_reward(node)
        else:
            offspring = self.transition.rollout(node)
            return self.get_objective_values_and_reward(offspring)

    def _backpropagate(self, node: Node, value: float, use_dummy_reward: bool):
        while node:
            node.observe(0 if use_dummy_reward else value)
            node = node.parent

    # implement
    def _generate_impl(self):
        node = self._selection()
        
        if node.depth > self.max_tree_depth:
            node.mark_as_terminal(freeze=self.freeze_terminal)  
        elif not node.children and node.n != 0:
            if not self._expand(node):
                node.mark_as_terminal(freeze=self.freeze_terminal)
                
        if node.is_terminal():
            if self.terminal_reward != "ignore":
                self._backpropagate(node, self.terminal_reward, False)
            return

        if not node.children:
            children = [node]
        else:
            children = node.sample_children(max_size=self.rollout_width, replace=self.allow_rollout_overlaps)
        
        parent_got_unfiltered_node = False
        for child in children:
            child_got_unfiltered_node = False
            for _ in range(self.n_rollouts):
                for _ in range(self.n_tries):
                    objective_values, reward = self._eval(child) # returns the child itself if terminal
                    if type(objective_values[0]) != str: # not filtered
                        break
                if type(objective_values[0]) != str: # not filtered
                    child_got_unfiltered_node = parent_got_unfiltered_node = True
                    self._backpropagate(child, reward, self.use_dummy_reward)
                elif self.filtered_reward[int(objective_values[0])] != "ignore":
                    self._backpropagate(child, self.filtered_reward[int(objective_values[0])], False)
            if self.remove_failed_child and not child_got_unfiltered_node:
                del child.parent.children[child.last_action]
        if self.all_filtered_reward != "ignore" and not parent_got_unfiltered_node:
            self._backpropagate(node, self.all_filtered_reward, False)
            self.logger.debug("All rollouts failed from: " + str(node))