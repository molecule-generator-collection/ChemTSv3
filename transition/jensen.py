from collections import OrderedDict
import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from filter import Filter
from node import CanonicalSMILESStringNode
from transition import TemplateTransition

class JensenTransition(TemplateTransition):
    """
    Ref: https://github.com/jensengroup/GB_GA/tree/master by Jan H. Jensen 2018
    """
    
    def __init__(self, base_chances=[0.15,0.14,0.14,0.14,0.14,0.14,0.15], check_size: bool=True, average_size: float=50.0, size_stdev: float=5.0, check_ring: bool=True, merge_duplicates: bool=True, record_actions: bool=False, filters: list[Filter]=None, top_p=None, logger=None):
        """
        Args:
            base_chances: chances of [insert_atom, change_bond_order, delete_cyclic_bond, add_ring, delete_atom, change_atom, append_atom]
            average_size: Used for the molecule size filter only if check_size is True.
            size_stdev: Used for the molecule size filter only if check_size is True.
            record_actions: If True, used smirks will be recorded as actions in child nodes.
        """
        self.base_chances = base_chances
        self.check_size = check_size
        self.average_size = average_size
        self.size_stdev = size_stdev
        self.check_ring = check_ring
        self.merge_duplicates = merge_duplicates
        self.record_actions = record_actions
        super().__init__(filters=filters, top_p=top_p, logger=logger)

    @staticmethod
    def delete_atom():
        choices = ['[*:1]~[D1]>>[*:1]', '[*:1]~[D2]~[*:2]>>[*:1]-[*:2]',
                    '[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]',
                    '[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]',
                    '[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]']
        p = [0.25,0.25,0.25,0.1875,0.0625]

        return choices, p
    
    @staticmethod
    def append_atom():
        BOs = ['single', 'double', 'triple']
        atom_lists = [['C','N','O','F','S','Cl','Br'], ['C','N','O'], ['C','N']]
        probs = [[1/7.0]*7, [1/3.0]*3, [1/2.0]*2]
        p_BO = [0.60, 0.35, 0.05]

        choices = []
        p = []

        for bo, atoms, atom_p in zip(BOs, atom_lists, probs):
            for a, prob in zip(atoms, atom_p):
                if bo == 'single':
                    smarts = f'[*;!H0:1]>>[*:1]-{a}'
                elif bo == 'double':
                    smarts = f'[*;!H0;!H1:1]>>[*:1]={a}'
                else: # triple
                    smarts = f'[*;H3:1]>>[*:1]#{a}'
                choices.append(smarts)
                p.append(p_BO[BOs.index(bo)] * prob)

        total = sum(p)
        p = [v/total for v in p]

        return choices, p
    
    @staticmethod
    def insert_atom():
        bond_opts = {
            "single": (["C", "N", "O", "S"], 0.60),
            "double": (["C", "N"], 0.35),
            "triple": (["C"], 0.05),
        }

        choices, p = [], []
        for BO, (atoms, p_bo) in bond_opts.items():
            p_element = p_bo / len(atoms)
            for a in atoms:
                if BO == "single":
                    smarts = f"[*:1]~[*:2]>>[*:1]{a}[*:2]"
                elif BO == "double":
                    smarts = f"[*;!H0:1]~[*:2]>>[*:1]={a}-[*:2]"
                else:  # triple
                    smarts = f"[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#{a}-[*:2]"
                choices.append(smarts)
                p.append(p_element)

        return choices, p

    @staticmethod
    def change_bond_order():
        choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]','[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
                    '[*:1]#[*:2]>>[*:1]=[*:2]','[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
        p = [0.45,0.45,0.05,0.05]

        return choices, p

    @staticmethod
    def delete_cyclic_bond():
        return ['[*:1]@[*:2]>>([*:1].[*:2])'], [1.0]

    @staticmethod
    def add_ring():
        choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
                    '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
                    '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
                    '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1'] 
        p = [0.05,0.05,0.45,0.45]
    
        return choices, p
    
    @staticmethod
    def change_atom():
        elements = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
        p_elem   = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

        choices, weights = [], []
        for (x, px), (y, py) in itertools.product(zip(elements, p_elem), zip(elements, p_elem)):
            if x == y:
                continue
            choices.append(f"[{x}:1]>>[{y}:1]")
            weights.append(px * py)

        total = sum(weights)
        p = [w / total for w in weights]

        return choices, p

    @staticmethod
    def ring_OK(mol):
        try:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
                return True

            ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

            cycle_list = mol.GetRingInfo().AtomRings() 
            max_cycle_length = max([ len(j) for j in cycle_list ])
            macro_cycle = max_cycle_length > 6

            double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

            return not ring_allene and not macro_cycle and not double_bond_in_small_ring
        except:
            return False

    def mol_OK(self, mol):
        try:
            target_size = self.size_stdev*np.random.randn() + self.average_size
            if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
                return True
            else:
                return False
        except:
            return False
    
    @staticmethod
    def try_sanitize(mol):
        try:
            Chem.SanitizeMol(mol)
            return True
        except:
            return False

    # implement
    def _next_nodes_impl(self, node: CanonicalSMILESStringNode) -> list[CanonicalSMILESStringNode]:
        try:
            mol = node.mol(use_cache=False)
        
            Chem.Kekulize(mol, clearAromaticFlags=True)
            rxn_smarts_list = 7*['']
            rxn_smarts_list[0] = self.insert_atom()
            rxn_smarts_list[1] = self.change_bond_order()
            rxn_smarts_list[2] = self.delete_cyclic_bond()
            rxn_smarts_list[3] = self.add_ring()
            rxn_smarts_list[4] = self.delete_atom()
            rxn_smarts_list[5] = self.change_atom()
            rxn_smarts_list[6] = self.append_atom()
            raw_result = [] # action, SMILES, raw_prob

            for i, (choices, probs) in enumerate(rxn_smarts_list):
                if self.base_chances[i] == 0:
                    continue
                for smarts, prob in zip(choices, probs):
                    new_prob = prob * self.base_chances[i]
                    action = smarts if self.record_actions else None
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    new_mol_trial = rxn.RunReactants((mol,))
                    
                    new_smis = []
                    for m in new_mol_trial:
                        m = m[0]
                        if self.try_sanitize(m) and (not self.check_size or self.mol_OK(m)) and (not self.check_ring or self.ring_OK(m)):
                            try:
                                smiles = Chem.MolToSmiles(m, canonical=True)
                                new_smis.append(smiles)
                            except:
                                continue
                            
                    for smiles in new_smis:
                        last_prob = new_prob * (1 / len(new_smis))
                        raw_result.append((action, smiles, last_prob))
            
            if self.merge_duplicates:
                raw_result = self.merge_duplicate_smiles(raw_result)
                        
            total = sum(prob for _, _, prob in raw_result)
            if total == 0:
                return []
            return [CanonicalSMILESStringNode(string=smiles, parent=node, last_action=a, last_prob=prob/total) for a, smiles, prob in raw_result]
        except:
            return []
    
    @staticmethod
    def merge_duplicate_smiles(tuples: list[tuple]) -> list[tuple]:
        smiles_dict = {}
        for action, smiles, prob in tuples:
            if smiles in smiles_dict:
                smiles_dict[smiles][1] += prob
            else:
                smiles_dict[smiles] = [action, prob]

        return [(action, smiles, prob) for smiles, (action, prob) in smiles_dict.items()]

