from rdkit.Chem import Mol, Descriptors, rdMolDescriptors
from filter import MolFilter

class LipinskiFilter(MolFilter):
    def __init__(self, rule_of: int = None, max_mol_weight = None, max_logP = None, max_hydrogen_bond_donors = None, max_hydrogen_bond_acceptors = None, max_rotatable_bonds = None):
        """
        Prioritize max_*** over rule_of value.
        """
        if not (rule_of is None or rule_of == 3 or rule_of == 5):
            raise AssertionError("rule_of must be either 5, 3, or None")
        
        self.max_hydrogen_bond_acceptors = self.max_mol_weight = self.max_rotatable_bonds = None
        if rule_of:
            self.max_mol_weight = 500 * rule_of
            if rule_of == 3:
                self.max_hydrogen_bond_acceptors = 3
                self.max_rotatable_bonds = 3
            elif rule_of == 5:
                self.max_hydrogen_bond_acceptors = 10
            
        for field in ["max_logP", "max_hydrogen_bond_donors"]:
            value = locals()[field]
            if value is None and rule_of is None:
                setattr(self, field, float("inf"))
            elif value is not None:
                setattr(self, field, value)
            else:
                setattr(self, field, rule_of)
            
        for field in ["max_hydrogen_bond_acceptors", "max_mol_weight", "max_rotatable_bonds"]:
            value = locals()[field]
            if value is None and getattr(self, field) is None:
                setattr(self, field, float("inf"))
            elif value is not None:
                setattr(self, field, value)
            
            
    #implement
    def mol_check(self, mol: Mol) -> bool:
        mol_weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
        logP = Descriptors.MolLogP(mol)
        n_hydrogen_bond_donors = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        n_hydrogen_bond_acceptors = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        n_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return (mol_weight <= self.max_mol_weight
                and logP <= self.max_logP
                and n_hydrogen_bond_donors <= self.max_hydrogen_bond_donors
                and n_hydrogen_bond_acceptors <= self.max_hydrogen_bond_acceptors
                and n_rotatable_bonds <= self.max_rotatable_bonds)