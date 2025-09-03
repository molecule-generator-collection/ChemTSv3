from rdkit import Chem
from rdkit.Chem import AllChem
from node import SMILESStringNode
from transition import TemplateTransition

class SMIRKSTransition(TemplateTransition):
    def __init__(self, smirks_path: str=None, weighted_smirks: list[tuple[str, float]]=None, without_Hs: bool=True, with_Hs: bool=False, kekulize=True, filters=None, top_p=None, record_actions=False):
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
        super().__init__(filters=filters, top_p=top_p)
                    
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
    def _next_nodes_impl(self, node: SMILESStringNode):
        try:
            initial_mol = node.mol(use_cache=False)
            if self.kekulize:
                Chem.Kekulize(initial_mol, clearAromaticFlags=True)
            if self.with_Hs:
                initial_mol_with_Hs = Chem.AddHs(initial_mol)
            generated_mols = []
            for smirks, weight in self.weighted_smirks:
                try:
                    rxn = AllChem.ReactionFromSmarts(smirks)
                    products = []
                    if self.without_Hs:
                        products += rxn.RunReactants((initial_mol,))
                    if self.with_Hs:
                        products += rxn.RunReactants((initial_mol_with_Hs,))
                    for ps in products:
                        for p in ps:
                            generated_mols.append((smirks, weight, p))
                except:
                    continue
                
            sum_weight = 0
            weights = {}
            actions = {}
            
            for smirks, weight, mol in generated_mols:
                try:
                    mol = Chem.RemoveHs(mol)
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    sum_weight += weight
                    if smiles not in weights:
                        weights[smiles] = weight
                        actions[smiles] = smirks
                    else:
                        weights[smiles] += weight
                        actions[smiles] += f" or {smirks}"
                except:
                    continue
                
            children = []
            for smiles in weights.keys():
                weight = weights[smiles]
                action = actions[smiles] if self.record_actions else None
                child = SMILESStringNode(string=smiles, parent=node, last_prob=weight/sum_weight, last_action=action)
                children.append(child)
                
            return children
        except:
            return []