from rdkit import Chem
from rdkit.Chem import AllChem
from node import SMILESStringNode
from transition import Transition

class SMIRKSTransition(Transition):
    def __init__(self, smirks_list_path: str=None, smirks_list: list[str]=None):
        """
        Args:
            smirks_list_path: Path to a .txt file containing SMIRKS patterns, one per line. Empty lines and text after '#' are ignored.
            smirks_list: A list of SMIRKS patterns.
        
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
            
    def load_smirks(self, path: str):
        self.smirks_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip() # remove comments and space
                if line:  # if not empty
                    self.smirks_list.append(line)
        
    # implement
    def transitions_with_probs(self, node: SMILESStringNode):
        initial_mol = node.mol()
        generated_mols = []
        for smirks in self.smirks_list:
            try:
                rxn = AllChem.ReactionFromSmarts(smirks)

                products = rxn.RunReactants((initial_mol,))
                for ps in products:
                    for p in ps:
                        generated_mols.append(p)
            except:
                continue
                
        unique_smiles = set()

        for mol in generated_mols:
            smiles = Chem.MolToSmiles(mol)
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)

        unique_smiles = list(unique_smiles)
        
        transitions = []
        for i, smiles in enumerate(unique_smiles):
            # TODO: set meaningful action labels
            node = SMILESStringNode(string=smiles, parent=node, last_prob=1/len(unique_smiles), last_action=i)
            transitions.append((i, node, 1/len(unique_smiles)))
            
        return transitions