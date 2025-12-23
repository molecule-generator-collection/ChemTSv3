from rdkit import Chem
from rdkit.Chem import AllChem
from chemtsv3.node import SMILESStringNode
from chemtsv3.transition import TemplateTransition

def apply_smirks(mol, smirks) -> list[str]:
    rxn = AllChem.ReactionFromSmarts(smirks)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    results = [] # list of SMILES
    for ps in rxn.RunReactants((mol,)):
        for p in ps:
            try:
                p = Chem.RemoveHs(p)
                smiles = Chem.MolToSmiles(p, canonical=True)
                results.append(smiles)
            except:
                continue
    results = list(set(results)) # remove duplicates
    return results

class ExampleTransition(TemplateTransition):
    def __init__(self, smirks_rules: list[str], top_p=None, filters=None, logger=None):
        self.smirks_rules = smirks_rules
        super().__init__(filters=filters, top_p=top_p, logger=logger) # Call __init__() of TemplateTransition to use those parameters
    
    # Define transition here
    def _next_nodes_impl(self, node: SMILESStringNode) -> list[SMILESStringNode]: # Input: initial node / Output: list of resulting nodes
        initial_mol = node.mol()
        smiles_list = []
        for smirks in self.smirks_rules:
            smiles_list += apply_smirks(initial_mol, smirks)
        smiles_list = list(set(smiles_list)) # Remove duplicates
        
        results = []
        for smiles in smiles_list:
            child_node = SMILESStringNode(string=smiles, parent=node, last_prob=1) # last_prob will be automatically normalized in TemplateTransition
            results.append(child_node)
            
        return results