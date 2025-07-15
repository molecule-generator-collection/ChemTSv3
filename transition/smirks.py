from rdkit import Chem
from rdkit.Chem import AllChem
from node import SMILESStringNode
from transition import Transition

class SMIRKSTransition(Transition):
    def __init__(self, smirks_list_path: str=None, smirks_list: list[str]=None, without_Hs: bool=True, with_Hs: bool=False, kekulize=True):
        """
        Args:
            smirks_list_path: Path to a .txt file containing SMIRKS patterns, one per line. Empty lines and text after '##' are ignored.
            smirks_list: A list of SMIRKS patterns.
            without_Hs: If True, SMIRKS reactions are applied to the molecule without explicit hydrogens. Defaults to True.
            with_Hs: If True, SMIRKS reactions are applied to the molecule with explicit hydrogens (via `Chem.AddHs`). Defaults to True.
        
        Raises:
            ValueError: If both or neither of 'smirks_list_path' and 'smirks_list' are specified.
        """
        if smirks_list_path is not None and smirks_list is not None:
            raise ValueError("Specify either 'smirks_list_path' or 'smirks_list', not both.")
        elif smirks_list_path is not None:
            self.load_smirks(smirks_list_path)
        elif smirks_list is not None:
            self.smirks_list = smirks_list
        else:
            raise ValueError("Specify either 'smirks_list_path' or 'smirks_list'.")
        if not without_Hs and not with_Hs:
            raise ValueError("Set one or both of 'check_no_Hs' or 'check_Hs' to True.")        

        self.without_Hs = without_Hs
        self.with_Hs = with_Hs
        self.kekulize = kekulize
            
    def load_smirks(self, path: str):
        self.smirks_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("##", 1)[0].strip() # remove comments and space
                if line:  # if not empty
                    self.smirks_list.append(line)
        
    # implement
    def transitions_with_probs(self, node: SMILESStringNode):
        initial_mol = node.mol(use_cache=False)
        if self.kekulize:
            Chem.Kekulize(initial_mol, clearAromaticFlags=True)
        if self.with_Hs:
            initial_mol_with_Hs = Chem.AddHs(initial_mol)
        generated_mols = []
        for smirks in self.smirks_list:
            try:
                rxn = AllChem.ReactionFromSmarts(smirks)
                products = []
                if self.without_Hs:
                    products += rxn.RunReactants((initial_mol,))
                if self.with_Hs:
                    products += rxn.RunReactants((initial_mol_with_Hs,))
                for ps in products:
                    for p in ps:
                        generated_mols.append((smirks, p))
            except:
                continue
                
        unique_smiles_set = set()
        unique_transitions = []

        for smirks, mol in generated_mols:
            try:
                mol = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol)
                if smiles not in unique_smiles_set:
                    unique_smiles_set.add(smiles)
                    unique_transitions.append((smirks, smiles))
            except:
                continue
        
        transitions = []
        for i, (smirks, smiles) in enumerate(unique_transitions):
            child = SMILESStringNode(string=smiles, parent=node, last_prob=1/len(unique_transitions), last_action=(smirks, i))
            transitions.append(((smirks, i), child, 1/len(unique_transitions)))
            
        return transitions