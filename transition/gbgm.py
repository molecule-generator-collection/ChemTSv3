import numpy as np
import os
import pickle
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from filter import Filter
from node import CanonicalSMILESStringNode
from transition import TemplateTransition

class GBGMTransition(TemplateTransition):
    """
    Ref: https://github.com/jensengroup/GB-GM/blob/master/ by Jan H. Jensen 2018
    GBGM paper: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c
    """
    
    def __init__(self, average_size: float=50.0, size_std: float=5.0, max_children: int=25, max_expansion_tries: int=5000, prob_ring_atom: float=0.63, prob_double=0.8, filters: list[Filter]=None, top_p=None, logger=None):
        """
        Args:
            average_size: Used for the molecule size filter only if check_size is True.
            size_std: Used for the molecule size filter only if check_size is True.
            prob_ring_atom: The probability of adding ring atom
            record_actions: If True, used smirks will be recorded as actions in child nodes.
        """
        self.average_size = average_size
        self.size_std = size_std
        self.max_children = max_children
        self.prob_ring_atom = prob_ring_atom
        self.prob_double = prob_double
        self.max_expansion_tries = max_expansion_tries
        
        self.rxn_smarts_make_ring = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gbgm/rs_make_ring.p")),"rb"))
        self.rxn_smarts_ring_list = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gbgm/rs_ring.p")),"rb"))
        self.p_ring = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gbgm/p_ring.p")),"rb"))
        self.p_ring = self.scale_p_ring(self.rxn_smarts_ring_list, self.p_ring, self.prob_double)
        self.p_make_ring = self.p_ring
        
        self.rxn_smarts_list = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gbgm/r_s1.p")),"rb"))
        self.p = pickle.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gbgm/p1.p")),"rb"))
        
        super().__init__(filters=filters, top_p=top_p, logger=logger)
    
    @staticmethod
    def scale_p_ring(rxn_smarts_ring_list, p_ring, new_prob_double):
        p_single = []
        p_double = []
        for smarts,p in zip(rxn_smarts_ring_list,p_ring):
            if '=' in smarts:
                p_double.append(p)
            else:
                p_single.append(p)
            
        prob_double, prob_single = sum(p_double), sum(p_single)
        scale_double = new_prob_double/prob_double
        scale_single = (1.0 - new_prob_double)/(1-prob_double)
        for i, smarts in enumerate(rxn_smarts_ring_list):
            if '=' in smarts:
                p_ring[i] *= scale_double
            else:
                p_ring[i] *= scale_single
            
        # print(scale_double, scale_single*prob_single, sum(p_ring))
        return p_ring
    
    @staticmethod
    def run_rxn(rxn_smarts, mol) -> Mol:
        new_mol_list = []
        patt = rxn_smarts.split('>>')[0]
        # work on a copy so an un-kekulized version is returned
        # if the molecule is not changed
        mol_copy = Chem.Mol(mol)
        try:
            Chem.Kekulize(mol_copy)
        except:
            pass
        if mol_copy.HasSubstructMatch(Chem.MolFromSmarts(patt)):
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            new_mols = rxn.RunReactants((mol_copy,))
            for new_mol in new_mols:
                try:
                    Chem.SanitizeMol(new_mol[0])
                    new_mol_list.append(new_mol[0])
                except:
                    pass
            if len(new_mol_list) > 0:
                new_mol = random.choice(new_mol_list) 
                return new_mol
            else:
                return mol
        else:
            return mol

    def add_atom(self, mol) -> tuple[Mol, str]:
        if np.random.random() < self.prob_ring_atom: # probability of adding ring atom
            rxn_smarts = np.random.choice(self.rxn_smarts_ring_list, p=self.p_ring)
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4,r5]')) or AllChem.CalcNumAliphaticRings(mol) == 0:
                rxn_smarts = np.random.choice(self.rxn_smarts_make_ring, p=self.p_make_ring)
                if np.random.random() < 0.036: # probability of starting a fused ring
                    rxn_smarts = rxn_smarts.replace("!", "")
        else:
            if mol.HasSubstructMatch(Chem.MolFromSmarts('[*]1=[*]-[*]=[*]-1')):
                rxn_smarts = '[r4:1][r4:2]>>[*:1]C[*:2]'
            else:
                rxn_smarts = np.random.choice(self.rxn_smarts_list, p=self.p)
            
        mol = self.run_rxn(rxn_smarts,mol)
        smiles = Chem.MolToSmiles(mol)

        return mol, smiles
    
    def sample_child(self, initial_mol, initial_smiles) -> tuple[Mol, str]:        
        for _ in range(10):
            mol, smiles = self.add_atom(initial_mol)
            if smiles != initial_smiles:
                break

        return mol, smiles

    # implement
    def _next_nodes_impl(self, node: CanonicalSMILESStringNode) -> list[CanonicalSMILESStringNode]:
        if node.has_reward():
            return []
        seen = set()
        children = []

        initial_mol = node.mol()
        initial_smiles = node.string        

        try_count = 0
        while len(children) < self.max_children and try_count < self.max_expansion_tries:
            try_count += 1
            mol, smiles = self.sample_child(initial_mol, initial_smiles)
            if not smiles in seen:
                seen.add(smiles)
                children.append((mol, smiles))
        
        results = []
        for i, (mol, smiles) in enumerate(children):
            if mol.GetNumAtoms() > self.size_std*np.random.randn() + self.average_size:
                s = smiles + node.eos
            else:
                s = smiles
            results.append(CanonicalSMILESStringNode(s, parent=node))
                
        return results