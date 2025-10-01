import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from node import CanonicalSMILESStringNode
from transition import TemplateTransition

class SMIRKSTransition(TemplateTransition):
    def __init__(self, smirks_path: str=None, weighted_smirks: list[tuple[str, float]]=None, without_Hs: bool=True, with_Hs: bool=False, kekulize=True, filters=None, top_p=None, logger=None, record_actions=True, output_dir: str=None):
        """
        Args:
            smirks_path: Path to a .txt file containing SMIRKS patterns, one per line. Empty lines and text after '##' are ignored. Optional weights can be specified after // (default: 1.0).
            weighted_smirks: A list of SMIRKS patterns.
            without_Hs: If True, SMIRKS reactions are applied to the molecule without explicit hydrogens. Defaults to True.
            with_Hs: If True, SMIRKS reactions are applied to the molecule with explicit hydrogens (via `Chem.AddHs`). Defaults to False.
        
        Raises:
            ValueError: If both or neither of 'smirks_path' and 'weighted_smirks' are specified.
        """
        if smirks_path is not None and weighted_smirks is not None:
            raise ValueError("Specify either 'smirks_path' or 'weighted_smirks', not both.")
        elif smirks_path is not None:
            self.load_smirks(smirks_path)
        elif weighted_smirks is not None:
            self.weighted_smirks = weighted_smirks
        else:
            raise ValueError("Specify either 'smirks_path' or 'weighted_smirks'.")
        if not without_Hs and not with_Hs:
            raise ValueError("Set one or both of 'check_no_Hs' or 'check_Hs' to True.")        

        self.without_Hs = without_Hs
        self.with_Hs = with_Hs
        self.kekulize = kekulize
        self.record_actions = record_actions
        self.output_dir = output_dir
        
        self.n = 0
        self.sum_delta_unfiltered = 0
        self.sum_delta_including_filtered = 0
        self.sum_deltas_unfiltered = {}
        self.sum_deltas_including_filtered = {}
        self.ns = {}
        self.n_filtered = {}
        
        super().__init__(filters=filters, top_p=top_p, logger=logger)
                    
    def load_smirks(self, path: str):
        self.weighted_smirks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("##", 1)[0].strip() # remove comments and space
                if not line:
                    continue
                if "//" in line:
                    smirks, weight_str = line.split("//", 1)
                    smirks = smirks.strip()
                    try:
                        weight = float(weight_str.strip())
                    except ValueError:
                        raise ValueError(f"Invalid weight: '{weight_str.strip()}' in line: {line}")
                else:
                    smirks = line
                    weight = 1.0  # default weight
                self.weighted_smirks.append((smirks, weight))
        
    # implement
    def _next_nodes_impl(self, node: CanonicalSMILESStringNode) -> list[CanonicalSMILESStringNode]:
        try:
            initial_mol = node.mol(use_cache=False)
            if self.kekulize:
                Chem.Kekulize(initial_mol, clearAromaticFlags=True)
            if self.with_Hs:
                initial_mol_with_Hs = Chem.AddHs(initial_mol)
            generated_smis = []
            for smirks, weight in self.weighted_smirks:
                try:
                    rxn = AllChem.ReactionFromSmarts(smirks)
                    if self.without_Hs:
                        for ps in rxn.RunReactants((initial_mol,)):
                            for p in ps:
                                try:
                                    p = Chem.RemoveHs(p)
                                    smiles = Chem.MolToSmiles(p, canonical=True)
                                    generated_smis.append((smirks, weight, smiles))
                                except:
                                    continue
                    if self.with_Hs:
                        for ps in rxn.RunReactants((initial_mol_with_Hs,)):
                            for p in ps:
                                try:
                                    p = Chem.RemoveHs(p)
                                    smiles = Chem.MolToSmiles(p, canonical=True)
                                    generated_smis.append((smirks, weight, smiles))
                                except:
                                    continue
                except:
                    continue
                
            sum_weight = 0
            weights = {}
            actions = {}
            
            for smirks, weight, smiles in generated_smis:
                try:
                    sum_weight += weight
                    if smiles not in weights:
                        weights[smiles] = weight
                        actions[smiles] = smirks
                    else:
                        weights[smiles] += weight
                        # actions[smiles] += f" or {smirks}"
                except:
                    continue
                
            children = []
            for smiles in weights.keys():
                weight = weights[smiles]
                action = actions[smiles] if self.record_actions else None
                child = CanonicalSMILESStringNode(string=smiles, parent=node, last_prob=weight/sum_weight, last_action=action)
                children.append(child)
                
            return children
        except:
            return []
        
    # override     
    def observe(self, node, objective_values: list[float], reward: float, filtered: bool):
        if self.record_actions is True:
            action = node.last_action
            if node.parent.reward is None:
                return
            dif = reward - node.parent.reward

            self.n += 1
            if not action in self.ns:
                self.sum_deltas_unfiltered[action] = 0
                self.sum_deltas_including_filtered[action] = 0
                self.ns[action] = 1
                self.n_filtered[action] = 0
            else:
                self.ns[action] += 1
            
            if not filtered:
                self.sum_deltas_unfiltered[action] += dif
                self.sum_deltas_including_filtered[action] += dif
                self.sum_delta_unfiltered += dif
                self.sum_delta_including_filtered += dif
            else:
                self.sum_deltas_including_filtered[action] += dif
                self.sum_delta_including_filtered += dif
                self.n_filtered[action] += 1
                
    # override
    def analyze(self):
        records = []
        for key, _ in self.weighted_smirks:
            if not key in self.ns:
                continue
            n = self.ns[key]
            avg_unfiltered = self.sum_deltas_unfiltered[key] / n
            avg_filtered = self.sum_deltas_including_filtered[key] / n
            filtered_count = self.n_filtered[key]

            records.append({
                "SMIRKS": key,
                "Evaluation count": n,
                "Average delta (unfiltered)": avg_unfiltered,
                "Average delta (with filtered)": avg_filtered,
                "Filtered count": filtered_count
            })

        df = pd.DataFrame(records)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "smirks_stats.csv")
        df.to_csv(output_path, index=False)

        self.logger.info(f"Saved SMIRKS stats to {output_path}")
        return super().analyze()