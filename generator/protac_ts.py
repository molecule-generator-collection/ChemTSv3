import logging
import os
from chemtsv3.filter import MolFilter
from chemtsv3.generator import MCTS
from chemtsv3.node import MolNode
from chemtsv3.reward import MolReward
from chemtsv3.transition import Transition
from chemtsv3.utils import CSVHandler, ListFilter, link_linker

class PROTACTS(MCTS):
    args_for_extra_filters = ["filters_for_linked_mols"]
        
    def __init__(self, *args, reward: MolReward, ligands_1: str | list[str], ligands_2: str | list[str], filters_for_linked_mols: list[MolFilter], aggregation_type: str="max", output_dir: str=None, logger: logging.Logger=None, **kwargs):
        """
        Args:
            ligands_1, ligands_2: Lists of ligand SMILES to be linked with generated linker molecules.
            filters_for_linked_mols: List of filters applied to the linked molecules (i.e., after attaching ligands), instead of the raw linker molecules.
            aggregation_type: How to aggregate objective values and rewards of the linked molecules into a single representative value for the generated linker.
                - "max": Selects the linked molecule with the highest reward, and uses its reward and the corresponding objective values.
                - "mean": Uses the mean of objective values and rewards over all linked molecules.
        """
        super().__init__(*args, reward=reward, output_dir=output_dir, logger=logger, **kwargs) # output_dir and logger are explicit for generator_from_conf()
        
        if isinstance(ligands_1, str):
            ligands_1 = [ligands_1]
        self.ligands_1 = ligands_1
        if isinstance(ligands_2, str):
            ligands_2 = [ligands_2]
        self.ligands_2 = ligands_2
        
        self.filters_for_linked_mols = filters_for_linked_mols
        self.aggregation_type = aggregation_type
        self.logger_for_linked_mols = self._make_logger_for_linked_mols()
        
    # override
    def _get_objective_values_and_reward(self, node: MolNode) -> tuple[list[float], float]:
        """
        1. Perform duplication checks and apply filters to the raw linker molecule.
        2. For each linked molecule, apply filters; if passed, compute the reward and record the result to a separate CSV file.
        3. Aggregate the objective values and rewards of all valid linked molecules into a single representative value for the generated linker.
        """
        pre_reward_checks_result = self._pre_reward_checks(node)
        if pre_reward_checks_result[0] is True: # excludes 1.0
            key = pre_reward_checks_result[1]
        else:
            return pre_reward_checks_result
        
        smiles = node.smiles()
        linked_smiles, linked_mols, linked_ligands = [], [], []
        for ligand_1 in self.ligands_1:
            for ligand_2 in self.ligands_2:
                try:
                    link_molecule_smi = link_linker([ligand_1, ligand_2], smiles, linker_type="smiles", output_type="smiles")
                    link_molecule_mol = link_linker([ligand_1, ligand_2], smiles, linker_type="smiles", output_type="mol")
                    linked_smiles.append(link_molecule_smi)
                    linked_mols.append(link_molecule_mol)
                    linked_ligands.append((ligand_1, ligand_2))
                except:
                    self.logger.debug(f"Failed to link: {smiles} with {ligand_1} and {ligand_2}")
                    
        objective_values_of_linked_mols = []
        rewards_of_linked_mols = []
        
        for linked_mol, ls in zip(linked_mols, linked_ligands):
            for filter in self.filters_for_linked_mols:
                filter_result = filter.mol_check(linked_mol)
                if filter_result is not True: # excludes 1.0
                    self.logger.debug(f"Linked molecule of {smiles} with {ls[0]} and {ls[1]} was filtered by {filter.__class__.__name__}")
                else:
                    o, r = self.reward.objective_values_and_reward_from_mol(linked_mol)
                    objective_values_of_linked_mols.append(o)
                    rewards_of_linked_mols.append(r)
                    # record to csv
                    row = [len(self.unique_keys) + 1, self.passed_time, key, ls[0], ls[1], r, *o]
                    self.logger_for_linked_mols.info(row)
        
        if not rewards_of_linked_mols:
            self.logger.debug(f"All linked molecules of {smiles} were filtered.")
            node.clear_cache()
            return ["-1"], self.filter_reward[-1]

        objective_values, reward = self.aggregate_values(objective_values_of_linked_mols, rewards_of_linked_mols)
        self._post_reward_side_effects(node, key, objective_values, reward)
        return objective_values, reward
    
    def aggregate_values(self, objective_values_list: list[list[float]], reward_list: list[float]) -> tuple[list[float], float]:
        """
        Aggregate objective values and rewards of linked molecules.
        """
        if len(self.ligands_1) == 1 and len(self.ligands_2) == 1:
            return objective_values_list[0], reward_list[0]
        if self.aggregation_type == "max":
            idx = max(range(len(reward_list)), key=lambda i: reward_list[i])
            return objective_values_list[idx], reward_list[idx]
        elif self.aggregation_type == "mean":
            n = len(objective_values_list)
            d = len(objective_values_list[0])
            mean_objectives = [sum(obj[i] for obj in objective_values_list) / n for i in range(d)]
            mean_reward = sum(reward_list) / n
            return mean_objectives, mean_reward
        else:
            raise ValueError(f"Invalid aggregation_type: {self.aggregation_type}")
        
    def _make_logger_for_linked_mols(self):
        path = os.path.join(self.output_dir(), "linked_mol_results.csv")
        
        logger = logging.getLogger(f"{__name__}.linked_mols")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        
        csv_handler = CSVHandler(path)
        csv_handler.addFilter(ListFilter())
        csv_handler.setLevel(logging.INFO)
        logger.addHandler(csv_handler)
        
        return logger
    
    # override
    def _write_csv_header(self):
        header = ["order", "time", "key", "ligand_1", "ligand_2", "reward"]
        header += self.reward.objective_names()
        self.logger_for_linked_mols.info(header)
        super()._write_csv_header()