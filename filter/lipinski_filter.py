from rdkit.Chem import Mol, Descriptors, rdMolDescriptors
from filter import MolFilter

class LipinskiFilter(MolFilter):
    def __init__(self, rule_of: int=None, max_weight=None, max_log_p=None, max_hydrogen_bond_donors=None, max_hydrogen_bond_acceptors=None, max_rotatable_bonds=None):
        """Prioritize max_*** over rule_of value."""
        if not (rule_of is None or rule_of == 3 or rule_of == 5):
            raise ValueError("rule_of must be either 5, 3, or None")
        
        if rule_of:
            self.max_weight = 100 * rule_of
            self.max_log_p = rule_of
            self.max_hydrogen_bond_donors = rule_of
            if rule_of == 3:
                self.max_hydrogen_bond_acceptors = 3
                self.max_rotatable_bonds = 3
            elif rule_of == 5:
                self.max_hydrogen_bond_acceptors = 10
            
        for field in ["max_weight", "max_log_p", "max_hydrogen_bond_donors", "max_hydrogen_bond_acceptors", "max_rotatable_bonds"]:
            value = locals()[field]
            if value is None and not hasattr(self, field):
                setattr(self, field, float("inf"))
            elif value is not None:
                setattr(self, field, value)
            
    # implement
    def mol_check(self, mol: Mol) -> bool:
        weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
        log_p = Descriptors.MolLogP(mol)
        n_hydrogen_bond_donors = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        n_hydrogen_bond_acceptors = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        n_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return (weight <= self.max_weight
                and log_p <= self.max_log_p
                and n_hydrogen_bond_donors <= self.max_hydrogen_bond_donors
                and n_hydrogen_bond_acceptors <= self.max_hydrogen_bond_acceptors
                and n_rotatable_bonds <= self.max_rotatable_bonds)