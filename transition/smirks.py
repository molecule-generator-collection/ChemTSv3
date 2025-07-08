from rdkit import Chem
from rdkit.Chem import AllChem
from node import SMILESStringNode
from transition import Transition

class SMIRKSTransition(Transition):
    def __init__(self, smirks_list_path: str=None, smirks_list: list[str]=None):
        """
        Args:
            smirks_list: A list of SMIRKS patterns, or the path to a .txt file containing them. If None, default patterns will be used.
        """
        if smirks_list_path is not None and smirks_list is not None:
            raise ValueError("Specify either 'smirks_list_path' or 'smirks_list', not both.")
        elif smirks_list_path is not None:
            self.load_smirks(smirks_list_path)
        elif smirks_list is not None:
            self.smirks_list = smirks_list
        else:
            self.smirks_list = [
        "[cH:1]>>[c:1]C", "[cH:1]>>[c:1]CC", "[cH:1]>>[c:1]F", "[cH:1]>>[c:1]Cl", "[cH:1]>>[c:1]O", "[cH:1]>>[c:1][N+](=O)[O-]", # benzene-derivative
        "[O:1][H]>>[O:1]C(C)=O", "[O:1][H]>>[O:1]C", "[O:1][H]>>[O:1]S(=O)(=O)c1ccc(C)cc1", # alcohol
        "[N:1]([H])[H]>>[N:1]C(C)=O", "[N:1]([H])[H]>>[N:1]C(=O)OC(C)(C)C", "[N:1]([H])[H]>>[N:1]S(=O)(=O)c1ccccc1", # amine
        "[C:1](=O)[O:2][H]>>[C:1](=O)[O:2]C", "[C:1](=O)[OH]>>[C:1](=O)[NH2]", "[C:1](=O)[OH]>>[C:1](=O)Cl", # carboxylic acid
        "[c:1][Br]>>[c:1]c1ccccc1", "[c:1][Cl]>>[c:1]N", # cross-coupling # "[c:1][I]>>[c:1]C#CH"?
        "[C:1]=[C:2]>>[C:1](O)[C:2](O)", "[C:1]=[C:2]>>[C:1](Br)[C:2](Br)", "[C:1](=O)[C:2]>>[C:1](=N[C:2])" # misc
        ]
            
    def load_smirks(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.smirks_list = [line.strip() for line in f if line.strip()]
        
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